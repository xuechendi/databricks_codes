import abc
from datetime import timedelta
from typing import Optional

from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject
from databricks.feature_store.entities.store_type import StoreType
from databricks.feature_store.online_store_spec import OnlineStoreSpec
from databricks.feature_store.protos.feature_store_serving_pb2 import (
    MySqlMetadata as ProtoMySqlMetadata,
    SqlServerMetadata as ProtoSqlServerMetadata,
    DynamoDbMetadata as ProtoDynamoDbMetadata,
    CosmosDbMetadata as ProtoCosmosDbMetadata,
)


class OnlineStoreAdditionalMetadata(_FeatureStoreObject):
    """
    Abstract class for defining online store additional metadata.
    """

    pass


class MySqlMetadata(OnlineStoreAdditionalMetadata):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    def to_proto(self):
        return ProtoMySqlMetadata(
            host=self.host,
            port=self.port,
        )

    @classmethod
    def from_proto(cls, mysql_metadata_proto: ProtoMySqlMetadata):
        return MySqlMetadata(mysql_metadata_proto.host, mysql_metadata_proto.port)


class SqlServerMetadata(OnlineStoreAdditionalMetadata):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    def to_proto(self):
        return ProtoSqlServerMetadata(
            host=self.host,
            port=self.port,
        )

    @classmethod
    def from_proto(cls, sql_server_metadata_proto: ProtoSqlServerMetadata):
        return SqlServerMetadata(
            sql_server_metadata_proto.host, sql_server_metadata_proto.port
        )


class DynamoDbMetadata(OnlineStoreAdditionalMetadata):
    def __init__(self, region: str, table_arn: str, ttl: Optional[timedelta]):
        self._region = region
        self._table_arn = table_arn
        self._ttl = ttl

    @property
    def region(self):
        return self._region

    @property
    def table_arn(self):
        return self._table_arn

    @property
    def ttl(self) -> Optional[timedelta]:
        return self._ttl

    def to_proto(self):
        proto_ttl = int(self.ttl.total_seconds()) if self.ttl else None
        return ProtoDynamoDbMetadata(
            region=self.region, table_arn=self.table_arn, ttl=proto_ttl
        )

    @classmethod
    def from_proto(cls, dynamodb_metadata_proto: ProtoDynamoDbMetadata):
        ttl = (
            timedelta(seconds=dynamodb_metadata_proto.ttl)
            if dynamodb_metadata_proto.HasField("ttl")
            else None
        )
        return DynamoDbMetadata(
            dynamodb_metadata_proto.region, dynamodb_metadata_proto.table_arn, ttl
        )


class CosmosDbMetadata(OnlineStoreAdditionalMetadata):
    def __init__(self, account_uri: str, container_uri: str):
        self._account_uri = account_uri
        self._container_uri = container_uri

    @property
    def account_uri(self):
        return self._account_uri

    @property
    def container_uri(self):
        return self._container_uri

    def to_proto(self):
        return ProtoCosmosDbMetadata(
            account_uri=self.account_uri, container_uri=self.container_uri
        )

    @classmethod
    def from_proto(cls, cosmosdb_metadata_proto):
        # The current MLR does not support Cosmos DB TTL. To ensure backwards incompatibility when future MLRs
        # introduce TTL support, throw here if an online store with a defined TTL is retrieved.
        if cosmosdb_metadata_proto.HasField("ttl"):
            raise ValueError(
                "Cosmos DB time-to-live (TTL) is not supported by this version of the databricks-feature-store library."
            )
        return CosmosDbMetadata(
            account_uri=cosmosdb_metadata_proto.account_uri,
            container_uri=cosmosdb_metadata_proto.container_uri,
        )


class OnlineStoreMetadata(abc.ABC):
    def __init__(self, online_store: OnlineStoreSpec, cloud_provider_unique_id: str):
        """
        Class representing the online store metadata that will be published to the Feature Catalog.
        Assumes that the online store spec is well formed (e.g. augmented with the relevant database/table names)
        """
        self._cloud = online_store.cloud
        self._store_type = online_store.store_type
        self._read_secret_prefix = online_store.read_secret_prefix
        self._write_secret_prefix = online_store.write_secret_prefix
        if (
            self._store_type == StoreType.AURORA_MYSQL
            or self._store_type == StoreType.MYSQL
        ):
            self._additional_metadata = MySqlMetadata(
                online_store.hostname, online_store.port
            )
        elif self._store_type == StoreType.SQL_SERVER:
            self._additional_metadata = SqlServerMetadata(
                online_store.hostname, online_store.port
            )
        elif self._store_type == StoreType.DYNAMODB:
            self._additional_metadata = DynamoDbMetadata(
                online_store.region, cloud_provider_unique_id, online_store.ttl
            )
        elif self._store_type == StoreType.COSMOSDB:
            self._additional_metadata = CosmosDbMetadata(
                account_uri=online_store.account_uri,
                container_uri=cloud_provider_unique_id,
            )
        self._online_table = online_store._get_online_store_name()

    @property
    def online_table(self):
        return self._online_table

    @property
    def cloud(self):
        return self._cloud

    @property
    def store_type(self):
        return self._store_type

    @property
    def read_secret_prefix(self):
        return self._read_secret_prefix

    @property
    def write_secret_prefix(self):
        return self._write_secret_prefix

    @property
    def additional_metadata(self) -> OnlineStoreAdditionalMetadata:
        return self._additional_metadata
