from typing import Optional

from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject
from databricks.feature_store.entities.cloud import Cloud
from databricks.feature_store.entities.online_store_metadata import (
    OnlineStoreAdditionalMetadata,
    MySqlMetadata,
    SqlServerMetadata,
    DynamoDbMetadata,
    CosmosDbMetadata,
)
from databricks.feature_store.entities.store_type import StoreType
from databricks.feature_store.protos.feature_store_serving_pb2 import (
    OnlineStoreDetailed as ProtoOnlineStoreDetailed,
)


class OnlineStoreDetailed(_FeatureStoreObject):
    """
    This class corresponds to the OnlineStoreDetailed proto message and describes an online store. It was introduced
    in v0.3.8, so previously deprecated fields (host, port) are unnecessary.
    """

    def __init__(
        self,
        cloud: Cloud,
        store_type: StoreType,
        name: str,
        creation_timestamp: int,
        last_updated_timestamp: int,
        creator_id: str,
        last_update_user_id: str,
        read_secret_prefix: Optional[str],
        additional_metadata: OnlineStoreAdditionalMetadata,
    ):

        """Initialize a OnlineStoreDetailed object."""
        self.cloud = cloud
        self.store_type = store_type
        self.name = name
        self.creation_timestamp = creation_timestamp
        self.last_updated_timestamp = last_updated_timestamp
        self.creator_id = creator_id
        self.last_update_user_id = last_update_user_id
        self.read_secret_prefix = read_secret_prefix
        self.additional_metadata = additional_metadata

    @classmethod
    def from_proto(cls, the_proto: ProtoOnlineStoreDetailed):
        """
        Return an OnlineStoreDetailed object from an OnlineStoreDetailed proto.

        Note: `repeated` proto fields are cast from `google.protobuf.pyext._message.RepeatedScalarContainer` to list.
        Additionally, relevant fields (e.g. additional_metadata) are converted to Python representations.
        """
        additional_metadata = None
        if the_proto.WhichOneof("additional_metadata") == "mysql_metadata":
            additional_metadata = MySqlMetadata.from_proto(the_proto.mysql_metadata)
        elif the_proto.WhichOneof("additional_metadata") == "sql_server_metadata":
            additional_metadata = SqlServerMetadata.from_proto(
                the_proto.sql_server_metadata
            )
        elif the_proto.WhichOneof("additional_metadata") == "dynamodb_metadata":
            additional_metadata = DynamoDbMetadata.from_proto(
                the_proto.dynamodb_metadata
            )
        elif the_proto.WhichOneof("additional_metadata") == "cosmosdb_metadata":
            additional_metadata = CosmosDbMetadata.from_proto(
                the_proto.cosmosdb_metadata
            )
        else:
            raise ValueError("Unsupported Store Type: " + str(the_proto))
        return cls(
            cloud=the_proto.cloud,
            store_type=the_proto.store_type,
            name=the_proto.name,
            creation_timestamp=the_proto.creation_timestamp,
            last_updated_timestamp=the_proto.last_updated_timestamp,
            creator_id=the_proto.creator_id,
            last_update_user_id=the_proto.last_update_user_id,
            read_secret_prefix=the_proto.read_secret_prefix,
            additional_metadata=additional_metadata,
        )
