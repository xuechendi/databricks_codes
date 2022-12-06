from databricks.feature_store.protos.feature_spec_pb2 import (
    FeatureTableInfo as ProtoFeatureTableInfo,
)


class FeatureTableInfo:
    def __init__(self, table_name: str, table_id: str):
        if not table_name:
            raise ValueError("table_name must be non-empty.")
        if not table_id:
            raise ValueError("table_id must be non-empty.")
        self._table_name = table_name
        self._table_id = table_id

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    @property
    def table_name(self):
        return self._table_name

    @property
    def table_id(self):
        return self._table_id

    @classmethod
    def from_proto(cls, feature_table_info_proto):
        return cls(
            table_name=feature_table_info_proto.table_name,
            table_id=feature_table_info_proto.table_id,
        )

    def to_proto(self):
        return ProtoFeatureTableInfo(
            table_name=self.table_name,
            table_id=self.table_id,
        )
