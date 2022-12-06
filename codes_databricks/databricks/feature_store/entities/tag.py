from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject


class Tag(_FeatureStoreObject):
    def __init__(self, key: str, value: str):
        self._key = key
        self._value = value

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> str:
        return self._value

    @classmethod
    def from_proto(cls, tag_proto):
        return cls(
            key=tag_proto.key,
            value=tag_proto.value,
        )
