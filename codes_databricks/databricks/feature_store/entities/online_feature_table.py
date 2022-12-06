from databricks.feature_store.entities.online_store_for_serving import (
    AbstractOnlineStoreForServing,
    OnlineStoreForServing,
    OnlineStoreForSageMakerServing,
)
from databricks.feature_store.entities.data_type import DataType
from databricks.feature_store.protos.feature_store_serving_pb2 import (
    OnlineFeatureTableForSageMakerServing as ProtoOnlineFeatureTableForSageMakerServing,
    PrimaryKeyDetails as ProtoPrimaryKeyDetails,
    FeatureDetails as ProtoFeatureDetails,
    TimestampKeyDetails as ProtoTimestampKeyDetails,
)
from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject

from typing import List, Optional


class PrimaryKeyDetails(_FeatureStoreObject):
    def __init__(self, name: str, data_type: DataType):
        self._name = name
        self._data_type = data_type

    @property
    def name(self):
        return self._name

    @property
    def data_type(self):
        return self._data_type

    def __str__(self):
        return "{" + self._name + "}"

    def __repr__(self):
        return str(self)


class TimestampKeyDetails(_FeatureStoreObject):
    def __init__(self, name: str, data_type: DataType):
        self._name = name
        self._data_type = data_type

    @property
    def name(self):
        return self._name

    @property
    def data_type(self):
        return self._data_type

    def __str__(self):
        return "{" + self._name + "}"

    def __repr__(self):
        return str(self)


class FeatureDetails(_FeatureStoreObject):
    def __init__(
        self, name: str, data_type: DataType, data_type_details: Optional[str] = None
    ):
        self._name = name
        self._data_type = data_type
        self._data_type_details = data_type_details

    @property
    def name(self):
        return self._name

    @property
    def data_type(self):
        return self._data_type

    @property
    def data_type_details(self):
        return self._data_type_details


class AbstractOnlineFeatureTable(_FeatureStoreObject):
    def __init__(
        self,
        feature_table_name: str,
        online_feature_table_name: str,
        online_store: AbstractOnlineStoreForServing,
        primary_keys: List[PrimaryKeyDetails],
        feature_table_id: str,
        features: List[FeatureDetails],
        timestamp_keys: List[TimestampKeyDetails],
    ):
        self._feature_table_name = feature_table_name
        self._online_feature_table_name = online_feature_table_name
        self._online_store = online_store
        self._primary_keys = primary_keys
        self._feature_table_id = feature_table_id
        self._features = features
        self._timestamp_keys = timestamp_keys

    @property
    def feature_table_name(self):
        return self._feature_table_name

    @property
    def online_feature_table_name(self):
        return self._online_feature_table_name

    @property
    def online_store(self):
        return self._online_store

    @property
    def primary_keys(self):
        return self._primary_keys

    @property
    def feature_table_id(self):
        return self._feature_table_id

    @property
    def features(self):
        return self._features

    @property
    def timestamp_keys(self):
        return self._timestamp_keys


class OnlineFeatureTable(AbstractOnlineFeatureTable):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @classmethod
    def from_proto(cls, the_proto):
        return cls(
            feature_table_name=the_proto.feature_table_name,
            online_feature_table_name=the_proto.online_feature_table_name,
            online_store=OnlineStoreForServing.from_proto(the_proto.online_store),
            primary_keys=[
                PrimaryKeyDetails(primary_key.name, primary_key.data_type)
                for primary_key in the_proto.primary_keys
            ],
            feature_table_id=the_proto.feature_table_id,
            features=[
                FeatureDetails(
                    feature.name, feature.data_type, feature.data_type_details
                )
                for feature in the_proto.features
            ],
            timestamp_keys=[
                TimestampKeyDetails(timestamp_key.name, timestamp_key.data_type)
                for timestamp_key in the_proto.timestamp_keys
            ],
        )


class OnlineFeatureTableForSageMakerServing(AbstractOnlineFeatureTable):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def to_proto(self) -> ProtoOnlineFeatureTableForSageMakerServing:
        proto = ProtoOnlineFeatureTableForSageMakerServing(
            feature_table_name=self.feature_table_name,
            online_feature_table_name=self.online_feature_table_name,
            feature_table_id=self._feature_table_id,
        )
        proto.online_store.CopyFrom(self.online_store.to_proto())
        for pk in self.primary_keys:
            proto.primary_keys.append(
                ProtoPrimaryKeyDetails(name=pk.name, data_type=pk.data_type)
            )
        for feat in self.features:
            proto.features.append(
                ProtoFeatureDetails(
                    name=feat.name,
                    data_type=feat.data_type,
                    data_type_details=feat.data_type_details,
                )
            )
        for timestamp_key in self.timestamp_keys:
            proto.timestamp_keys.append(
                ProtoTimestampKeyDetails(
                    name=timestamp_key.name, data_type=timestamp_key.data_type
                )
            )
        return proto

    @classmethod
    def from_proto(cls, the_proto):
        return cls(
            feature_table_name=the_proto.feature_table_name,
            online_feature_table_name=the_proto.online_feature_table_name,
            online_store=OnlineStoreForSageMakerServing.from_proto(
                the_proto.online_store
            ),
            primary_keys=[
                PrimaryKeyDetails(primary_key.name, primary_key.data_type)
                for primary_key in the_proto.primary_keys
            ],
            feature_table_id=the_proto.feature_table_id,
            features=[
                FeatureDetails(
                    feature.name, feature.data_type, feature.data_type_details
                )
                for feature in the_proto.features
            ],
            timestamp_keys=[
                TimestampKeyDetails(timestamp_key.name, timestamp_key.data_type)
                for timestamp_key in the_proto.timestamp_keys
            ],
        )
