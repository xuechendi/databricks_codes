from databricks.feature_store.api.proto.feature_catalog_pb2 import (
    KeySpec as ProtoKeySpec,
)
from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject


class KeySpec(_FeatureStoreObject):
    """
    Key spec for primary keys, partition keys, and features.
    Encodes name and data type for the feature.
    """

    def __init__(self, name: str, data_type: str, data_type_details: str = None):
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

    @classmethod
    def from_proto(cls, key_spec_proto):
        return cls(
            name=key_spec_proto.name,
            data_type=key_spec_proto.data_type,
            data_type_details=key_spec_proto.data_type_details,
        )

    def to_proto(self):
        return ProtoKeySpec(
            name=self.name,
            data_type=self.data_type.upper(),
            data_type_details=self.data_type_details,
        )

    def __eq__(self, other):
        if isinstance(other, KeySpec):
            return dict(self) == dict(other)
        return False
