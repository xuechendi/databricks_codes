from databricks.feature_store.protos.feature_store_serving_pb2 import (
    DataType as ProtoDataType,
)
from databricks.feature_store.entities._proto_enum_entity import _ProtoEnumEntity

from typing import Any
from pyspark.sql.types import ArrayType, MapType, DecimalType, DataType
import json
import re


class DataType(_ProtoEnumEntity):
    """Online store types."""

    INTEGER = ProtoDataType.Value("INTEGER")
    FLOAT = ProtoDataType.Value("FLOAT")
    BOOLEAN = ProtoDataType.Value("BOOLEAN")
    STRING = ProtoDataType.Value("STRING")
    DOUBLE = ProtoDataType.Value("DOUBLE")
    LONG = ProtoDataType.Value("LONG")
    TIMESTAMP = ProtoDataType.Value("TIMESTAMP")
    DATE = ProtoDataType.Value("DATE")
    SHORT = ProtoDataType.Value("SHORT")
    ARRAY = ProtoDataType.Value("ARRAY")
    MAP = ProtoDataType.Value("MAP")
    BINARY = ProtoDataType.Value("BINARY")
    DECIMAL = ProtoDataType.Value("DECIMAL")

    _FIXED_DECIMAL = re.compile("decimal\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)")

    @classmethod
    def _enum_type(cls) -> Any:
        return ProtoDataType

    @classmethod
    def from_spark_type(cls, spark_type):
        return cls.from_string(spark_type.typeName())

    @classmethod
    def top_level_type_supported(cls, spark_type: DataType) -> bool:
        """
        Checks whether the provided Spark data type is supported by Feature Store, only considering
        the top-level type for nested data types.

        Details on nested types:
          ArrayType: The elementType is not checked. Will return True.
          MapType: The keyType and valueType are not checked. Will return True.
          StructType: Not supported by Feature Store. Will return False.
        """
        cls.init()
        return spark_type.typeName().upper() in cls._STRING_TO_ENUM

    @classmethod
    def to_complex_spark_type(cls, json_value):
        """
        Constructs a complex Spark DataType from its compact JSON representation.

        Examples:
            - Input: '"decimal(1,2)"'
              Output: DecimalType(1,2)
            - Input: '{"containsNull":false,"elementType":"integer","type":"array"}'
              Output: ArrayType(IntegerType,false)
            - Input: '{"keyType":"integer","type":"map","valueContainsNull":True,"valueType":"integer"}'
              Output: MapType(IntegerType,IntegerType,true)
        """
        if not json_value:
            raise ValueError("Empty JSON value cannot be converted to Spark DataType")

        json_data = json.loads(json_value)
        if not isinstance(json_data, dict):
            # DecimalType does not have fromJson() method
            if json_value == "decimal":
                return DecimalType()
            if cls._FIXED_DECIMAL.match(json_data):
                m = cls._FIXED_DECIMAL.match(json_data)
                return DecimalType(int(m.group(1)), int(m.group(2)))

        if json_data["type"].upper() == cls.to_string(cls.ARRAY):
            return ArrayType.fromJson(json_data)

        if json_data["type"].upper() == cls.to_string(cls.MAP):
            return MapType.fromJson(json_data)

        else:
            raise ValueError(
                f"Spark type {json_data['type']} cannot be converted to a complex Spark DataType"
            )
