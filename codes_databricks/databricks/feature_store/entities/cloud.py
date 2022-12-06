from databricks.feature_store.protos.feature_store_serving_pb2 import (
    Cloud as ProtoCloud,
)
from databricks.feature_store.entities._proto_enum_entity import _ProtoEnumEntity

from typing import Any


class Cloud(_ProtoEnumEntity):
    """Cloud types."""

    AWS = ProtoCloud.Value("AWS")
    AZURE = ProtoCloud.Value("AZURE")
    GCP = ProtoCloud.Value("GCP")

    @classmethod
    def _enum_type(cls) -> Any:
        return ProtoCloud
