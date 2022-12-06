from databricks.feature_store.entities.column_info import ColumnInfo
from databricks.feature_store.protos.feature_spec_pb2 import (
    SourceDataColumnInfo as ProtoSourceDataColumnInfo,
)


class SourceDataColumnInfo(ColumnInfo):
    def __init__(self, name: str):
        if not name:
            raise ValueError("name must be non-empty.")
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def output_name(self):
        return self._name

    @classmethod
    def from_proto(cls, source_data_column_info_proto):
        return cls(name=source_data_column_info_proto.name)

    def to_proto(self):
        return ProtoSourceDataColumnInfo(name=self._name)
