from databricks.feature_store.entities.column_info import ColumnInfo
from databricks.feature_store.protos.feature_spec_pb2 import (
    FeatureColumnInfo as ProtoFeatureColumnInfo,
)
from typing import List


class FeatureColumnInfo(ColumnInfo):
    def __init__(
        self,
        table_name: str,
        feature_name: str,
        lookup_key: List[str],
        output_name: str,
        timestamp_lookup_key: List[str] = [],
    ):
        if not table_name:
            raise ValueError("table_name must be non-empty.")
        if not feature_name:
            raise ValueError("feature_name must be non-empty.")
        if not isinstance(lookup_key, list):
            raise ValueError("lookup_key must be a list.")
        if not lookup_key or "" in lookup_key or None in lookup_key:
            raise ValueError("lookup_key must be non-empty.")
        if not output_name:
            raise ValueError("output_name must be non-empty.")
        if not isinstance(timestamp_lookup_key, list):
            raise ValueError("timestamp_lookup_key must be a list.")

        self._table_name = table_name
        self._feature_name = feature_name
        self._lookup_key = lookup_key
        self._output_name = output_name
        self._timestamp_lookup_key = timestamp_lookup_key

    @property
    def table_name(self):
        return self._table_name

    @property
    def lookup_key(self):
        return self._lookup_key

    @property
    def feature_name(self):
        return self._feature_name

    @property
    def output_name(self):
        return self._output_name

    @property
    def timestamp_lookup_key(self):
        return self._timestamp_lookup_key

    @classmethod
    def from_proto(cls, feature_column_info_proto):
        return cls(
            table_name=feature_column_info_proto.table_name,
            feature_name=feature_column_info_proto.feature_name,
            lookup_key=list(feature_column_info_proto.lookup_key),
            output_name=feature_column_info_proto.output_name,
            timestamp_lookup_key=list(feature_column_info_proto.timestamp_lookup_key),
        )

    def to_proto(self):
        return ProtoFeatureColumnInfo(
            table_name=self.table_name,
            feature_name=self.feature_name,
            lookup_key=self.lookup_key,
            output_name=self.output_name,
            timestamp_lookup_key=self.timestamp_lookup_key,
        )
