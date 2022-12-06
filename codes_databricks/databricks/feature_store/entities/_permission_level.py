from databricks.feature_store.api.proto.feature_catalog_pb2 import (
    PermissionLevel as ProtoPermissionLevel,
)
from databricks.feature_store.entities._proto_enum_entity import _ProtoEnumEntity

from typing import Any


class _PermissionLevel(_ProtoEnumEntity):
    """Permission Levels."""

    _CAN_MANAGE = ProtoPermissionLevel.Value("CAN_MANAGE")
    _CAN_EDIT_METADATA = ProtoPermissionLevel.Value("CAN_EDIT_METADATA")
    _CAN_VIEW_METADATA = ProtoPermissionLevel.Value("CAN_VIEW_METADATA")

    @classmethod
    def _enum_type(cls) -> Any:
        return ProtoPermissionLevel

    @staticmethod
    def can_write_to_catalog(permission_level):
        return permission_level in [
            _PermissionLevel._CAN_MANAGE,
            _PermissionLevel._CAN_EDIT_METADATA,
        ]
