""" Defines the PublishDynamoDBEngine class, which defines the operations to be performed on
DynamoDB engine for publishing an offline feature table to online DynamoDB store.
"""

import abc
from typing import List, Dict, Optional

from botocore.exceptions import ClientError
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.types import (
    StructType,
)

from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec
from databricks.feature_store.utils.dynamodb_type_utils import (
    get_timestamp_key_type,
    BASIC_DATA_TYPE_CONVERTERS,
    COMPLEX_DATA_TYPE_CONVERTERS,
    get_type_converter,
)
from databricks.feature_store.utils.dynamodb_utils import (
    get_dynamodb_resource,
    to_dynamodb_primary_key,
    PRIMARY_KEY_SCHEMA,
    PRIMARY_KEY_ATTRIBUTE_NAME_VALUE,
    ATTRIBUTE_NAME,
    KEY_TYPE,
    ATTRIBUTE_TYPE,
    DYNAMODB_STRING_TYPE,
    RANGE,
    TTL_KEY_ATTRIBUTE_NAME_VALUE,
)


class PublishDynamoDBEngine(abc.ABC):
    DYNAMODB_BILLING_MODE = "PAY_PER_REQUEST"

    def __init__(self, online_store: AmazonDynamoDBSpec, spark_session: SparkSession):
        self.spark_session = spark_session
        self.access_key_id = online_store._lookup_access_key_id_with_write_permissions()
        self.secret_access_key = (
            online_store._lookup_secret_access_key_with_write_permissions()
        )
        self.session_token = online_store._lookup_session_token_with_write_permissions()
        self.region = online_store.region
        self.table_name = online_store.table_name
        self.ttl = online_store.ttl
        self._initialize_dynamodb_client_and_resource()

    def _initialize_dynamodb_client_and_resource(self):
        """
        Initializes DynamoDB client and resource objects to perform stateless operations on DynamoDB.
        """
        self._dynamodb_resource = get_dynamodb_resource(
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            region=self.region,
            session_token=self.session_token,
        )
        self._dynamodb_client = self._dynamodb_resource.meta.client

    def close(self):
        """
        Performs any close operations on the DynamoDB connections. DynamoDB connections are
        stateless http connections and hence does not need to be closed.
        :return:
        """
        pass

    def get_cloud_provider_unique_id(self) -> Optional[str]:
        """
        Returns cloud provider unique ID if the table exists, None otherwise
        """
        try:
            return self._dynamodb_resource.Table(self.table_name).table_arn
        except ClientError as ce:
            if ce.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            else:
                raise ce

    def create_empty_table(
        self,
        schema: StructType,
        primary_keys: List[str],
        timestamp_keys: List[str],
    ) -> str:
        """
        Creates an empty online table with the expected schema and configurations.
        (e.g. combining PKs, dropping the timestamp key, enabling ttl)
        """
        # Check: We only support a single timestamp key in DynamoDB.
        # TODO: ML-19665 add similar check in backend RPC validation.
        # TODO (ML-22021): move this validation to FeatureStoreClient.publish
        if len(timestamp_keys) > 1:
            raise ValueError(
                "Only one timestamp key is supported in DynamoDB online store."
            )

        # Check: The timestamp keys and publish mode are consistent.
        # DynamoDB tables not in window publish mode should not be created with timestamp keys,
        # as timestamp keys are logically mapped to and used in the DynamoDB composite PK RANGE column.
        if not self.is_timeseries_window_publish:
            timestamp_keys = []

        # Create the table. All validations should be prior.
        key_schema = self._get_key_schema(primary_keys, timestamp_keys)
        attribute_definitions = self._get_attribute_definitions(
            schema, primary_keys, timestamp_keys
        )
        self._dynamodb_client.create_table(
            TableName=self.table_name,
            KeySchema=key_schema,
            AttributeDefinitions=attribute_definitions,
            BillingMode=self.DYNAMODB_BILLING_MODE,
        )
        table = self._dynamodb_resource.Table(self.table_name)
        table.wait_until_exists()
        # Enable TTL attribute for this table if required.
        if self.is_timeseries_window_publish:
            self._dynamodb_client.update_time_to_live(
                TableName=self.table_name,
                TimeToLiveSpecification={
                    "Enabled": True,
                    "AttributeName": TTL_KEY_ATTRIBUTE_NAME_VALUE,
                },
            )
        # re-fetch table resource from boto3 to get the table ARN
        return self.get_cloud_provider_unique_id()

    @staticmethod
    def dynamodb_bulk_writes(
        table_name,
        df,
        primary_keys,
        access_key_id,
        secret_access_key,
        region,
        session_token,
    ):
        """
        Static method to make writes to DynamoDB using Spark foreachPatition operation.
        This method is static to avoid transferring any object state to worker. Passing object state
        fails since not all member variables are seriaizable.
        :param df:
        :param table_name:
        :param primary_keys:
        :param access_key_id:
        :param secret_access_key:
        :param region:
        :param session_token:
        :return:
        """
        schema = df.schema
        unsupported_data_types = [
            (field.name, field.dataType)
            for field in schema.fields
            if type(field.dataType) not in BASIC_DATA_TYPE_CONVERTERS
            and type(field.dataType) not in COMPLEX_DATA_TYPE_CONVERTERS
        ]

        if unsupported_data_types:
            plural = len(unsupported_data_types) > 1
            missing_cols_str = ", ".join(
                [
                    f"\n\t- {feat_name} (type: {feat_type})"
                    for (feat_name, feat_type) in unsupported_data_types
                ]
            )
            raise ValueError(
                f"Unsupported data types for DynamoDB publish. Feature{'s' if plural else ''}: {missing_cols_str}"
            )

        def row_to_items(row):
            item = {}
            primary_keys_list = []
            for index in range(len(row)):
                data_type = schema[index].dataType
                if schema[index].name in primary_keys:
                    primary_keys_list.append(
                        get_type_converter(data_type).to_dynamodb(row[index])
                    )
                else:
                    item[schema[index].name] = get_type_converter(
                        data_type
                    ).to_dynamodb(row[index])
            return {**item, **to_dynamodb_primary_key(primary_keys_list)}

        def dynamodb_bulk_write(rows):
            dynamodb_resource = get_dynamodb_resource(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                region=region,
                session_token=session_token,
            )
            table = dynamodb_resource.Table(table_name)

            with table.batch_writer() as batch:
                # DynamoDB batch writer handles chunking, buffering and retrying automatically.
                for row in rows:
                    batch.put_item(Item=row_to_items(row))

        df.foreachPartition(dynamodb_bulk_write)

    def write_table(
        self,
        df,
        schema: StructType,
        primary_keys: List[str],
        timestamp_keys: List[str],
    ):
        """
        Generates the expected schema and writes to an online table.
        (e.g. combining PKs, writing online-safe data types, generating ttl)
        """
        self.dynamodb_bulk_writes(
            self.table_name,
            df,
            primary_keys,
            self.access_key_id,
            self.secret_access_key,
            self.region,
            self.session_token,
        )

    def _get_key_schema(
        self, primary_keys: List[str], timestamp_keys: List[str]
    ) -> List[Dict[str, str]]:
        if not len(primary_keys):
            return []
        key_schema = [PRIMARY_KEY_SCHEMA]
        if len(timestamp_keys):
            key_schema.append({ATTRIBUTE_NAME: timestamp_keys[0], KEY_TYPE: RANGE})
        return key_schema

    def _get_attribute_definitions(
        self, schema: StructType, primary_keys: List[str], timestamp_keys: List[str]
    ) -> List[Dict[str, str]]:
        if not len(primary_keys):
            return []
        attribute_definitions = [
            {
                ATTRIBUTE_NAME: PRIMARY_KEY_ATTRIBUTE_NAME_VALUE,
                ATTRIBUTE_TYPE: DYNAMODB_STRING_TYPE,
            }
        ]
        if len(timestamp_keys):
            timestamp_key = timestamp_keys[0]
            timestamp_key_type = None
            for field in schema.fields:
                if field.name == timestamp_key:
                    timestamp_key_type = get_timestamp_key_type(field.dataType)
            attribute_definitions.append(
                {
                    ATTRIBUTE_NAME: timestamp_key,
                    ATTRIBUTE_TYPE: timestamp_key_type,
                }
            )

        return attribute_definitions

    def generate_df_with_ttl_if_required(self, df, timestamp_keys: List[str]):
        """
        Convert the timestamp column to epoch time in seconds per DynamoDB TTL expectations if TTL is defined.
        No buffer is required when generating the TTL column since the TTL is the event timestamp + the TTL duration.
        Both the event timestamp (from unix_timestamp) and DynamoDB are expected to be in UTC.
        See https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/time-to-live-ttl-before-you-start.html
        """
        if self.is_timeseries_window_publish:
            ttl_seconds = int(self.ttl.total_seconds())
            ttl_column = unix_timestamp(df[timestamp_keys[0]]) + ttl_seconds
            df = df.withColumn(TTL_KEY_ATTRIBUTE_NAME_VALUE, ttl_column)
        return df

    @property
    def is_timeseries_window_publish(self):
        """
        Determine if this publish engine will publish a window of a time series dataframe.
        This is currently equivalent to if TTL is defined in the online store spec.
        """
        return self.ttl is not None
