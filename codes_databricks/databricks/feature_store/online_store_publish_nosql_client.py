import time
from datetime import timedelta
from typing import List, Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp

from databricks.feature_store.constants import OVERWRITE
from databricks.feature_store.online_store_spec import (
    OnlineStoreSpec,
    AmazonDynamoDBSpec,
    AzureCosmosDBSpec,
)
from databricks.feature_store.online_store_publish_client import (
    OnlineTable,
    OnlineStorePublishClient,
    is_nosql_spec,
)
from databricks.feature_store.publish_engine import (
    PublishDynamoDBEngine,
    PublishCosmosDBEngine,
)
from databricks.feature_store.utils.publish_utils import get_latest_snapshot


def generate_nosql_engine(online_store_spec, spark_session):
    if isinstance(online_store_spec, AmazonDynamoDBSpec):
        return PublishDynamoDBEngine(online_store_spec, spark_session)
    elif isinstance(online_store_spec, AzureCosmosDBSpec):
        return PublishCosmosDBEngine(online_store_spec, spark_session)


class OnlineStorePublishNoSqlClient(OnlineStorePublishClient):
    def __init__(self, online_store: OnlineStoreSpec):
        if not is_nosql_spec(online_store):
            raise ValueError(f"Unexpected online store type {type(online_store)}")
        spark_session = SparkSession.builder.appName(
            "feature_store.nosql_client"
        ).getOrCreate()
        self.nosql_engine = generate_nosql_engine(online_store, spark_session)

    def get_or_create_online_table(self, df, primary_keys, timestamp_keys):
        cloud_provider_unique_id = self.nosql_engine.get_cloud_provider_unique_id()
        if not cloud_provider_unique_id:
            cloud_provider_unique_id = self.nosql_engine.create_empty_table(
                df.schema, primary_keys, timestamp_keys
            )
            new_table = True
        else:
            new_table = False
        return OnlineTable(
            self.nosql_engine.table_name, cloud_provider_unique_id, new_table
        )

    def _publish_merge_range_query(
        self,
        df,
        primary_keys: List[str],
        timestamp_keys: List[str],
        lookback_window: timedelta,
    ):
        """
        Helper to publish a window of a time series dataframe to the Online Store (used with QueryMode: RANGE_QUERY).
        The timestamp key should be defined as part of the logical primary key
        (e.g. the range key in DynamoDB, id property for Cosmos DB).
        """
        # Filter the df to all data since the start of the lookback window,
        # adding a day to the window to be safe as the database will automatically remove expired records.
        utc_timestamp = time.time()
        lookback_with_buffer = (
            lookback_window.total_seconds() + timedelta(days=1).total_seconds()
        )
        df = df.filter(
            unix_timestamp(df[timestamp_keys[0]]) > utc_timestamp - lookback_with_buffer
        )

        # Generate TTL column for the online store if required
        df = self.nosql_engine.generate_df_with_ttl_if_required(df, timestamp_keys)
        self.nosql_engine.write_table(df, df.schema, primary_keys, timestamp_keys)

    def _publish_merge_primary_keys_lookup(
        self, df, primary_keys: List[str], timestamp_keys: List[str]
    ):
        """
        Helper to publish a dataframe to the Online Store. For time series dataframes, this will publish the latest snapshot.
        The timestamp key should not be a part of the logical primary key:
        (e.g. the range key in DynamoDB, id property for Cosmos DB).
        """
        # TODO (ML-21776): Add test coverage for NoSQL snapshot tables
        if timestamp_keys:
            df = get_latest_snapshot(df, primary_keys, timestamp_keys)
        # For time series tables, `get_latest_snapshot` will retrieve the latest value for each primary key.
        # Because of this deduplication, timestamp keys are not required and instead treated as a regular feature.
        self.nosql_engine.write_table(df, df.schema, primary_keys, [])

    def _publish_overwrite(
        self,
        df,
        primary_keys: List[str],
        timestamp_keys: List[str],
        lookback_window: Optional[timedelta],
    ):
        raise NotImplementedError(f'Publish mode "{OVERWRITE}" is not supported.')

    def _publish_merge(
        self,
        df,
        primary_keys: List[str],
        timestamp_keys: List[str],
        lookback_window: Optional[timedelta],
    ):
        """
        Writes all the data from df into the Online Store. Existing data will be merged.
        """
        # If the lookback window is defined, we publish a range of data for use in QueryMode RANGE_QUERY.
        # Otherwise, we publish the data as-is for non-time series feature tables and the latest snapshot for time series
        # feature tables, both of which will be used in QueryMode PRIMARY_KEY_LOOKUP.
        if lookback_window is not None:
            self._publish_merge_range_query(
                df, primary_keys, timestamp_keys, lookback_window
            )
        else:
            self._publish_merge_primary_keys_lookup(df, primary_keys, timestamp_keys)

    def close(self):
        """
        Closes the nosql engine connection.
        :return:
        """
        self.nosql_engine.close()
