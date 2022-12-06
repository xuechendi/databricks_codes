""" Defines the type conversion classes from pyspark to DynamoDb compatible python types. These
 utils are used for converting from pyspark types in offline feature tables to DynamoDb compatible
  python types before publishing to DynamoDb.
  DD: go/dynamodb-fs-dd
"""

import calendar
import datetime
from abc import ABC, abstractmethod
from decimal import Decimal

from pyspark.sql.types import (
    DataType,
    BooleanType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    StringType,
    BinaryType,
    DateType,
    TimestampType,
    DecimalType,
    ArrayType,
    MapType,
)

from databricks.feature_store.utils.converter_utils import return_if_none
from databricks.feature_store.utils.dynamodb_utils import (
    DYNAMODB_STRING_TYPE,
    DYNAMODB_NUMBER_TYPE,
)

MICROSECONDS = 1000000


class DynamoDbPySparkTypeConverter(ABC):
    @staticmethod
    @abstractmethod
    def to_dynamodb(value):
        raise NotImplementedError

    @staticmethod
    def to_pyspark(value):
        # DynamoDB -> PySpark types is not required because we never need to read back values from
        # DynamoDb in Feature Store compute client, only lookup client.
        raise NotImplementedError


class DynamoDbPySparkBooleanTypeConverter(DynamoDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_dynamodb(value: bool) -> int:
        if value:
            return 1
        else:
            return 0


class DynamoDbPySparkShortIntLongTypeConverter(DynamoDbPySparkTypeConverter):
    """
    DynamoDB converter for Spark's Short, Integer, and Long data types. These data types are all mapped to DynamoDB's
    number type (which can contain all three losslessly), so type specific converters are not required.
    """

    @staticmethod
    @return_if_none
    def to_dynamodb(value: int) -> int:
        return int(value)


class DynamoDbPySparkFloatDoubleTypeConverter(DynamoDbPySparkTypeConverter):
    """
    DynamoDB converter for Spark's Float and Double data types. These data types are all mapped to DynamoDB's
    number type (which can contain both losslessly), so type specific converters are not required.
    """

    @staticmethod
    @return_if_none
    def to_dynamodb(value: float) -> Decimal:
        return Decimal(str(value))


class DynamoDbPySparkStringTypeConverter(DynamoDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_dynamodb(value: str) -> str:
        return str(value)


class DynamoDbPySparkTimestampTypeConverter(DynamoDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_dynamodb(value: datetime.datetime) -> int:
        # datetime.timestamp function handles the timestamp conversion to UTC.
        # We store the timestamp as epoch time in microseconds in UTC in DynamoDB.
        # https://docs.python.org/3/library/datetime.html#datetime.datetime.timestamp
        return int(value.timestamp() * MICROSECONDS)


class DynamoDbPySparkDateTypeConverter(DynamoDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_dynamodb(value: datetime.date) -> int:
        # calendar.timegm function handles the timestamp conversion to UTC.
        # We store the date as epoch time in seconds in UTC in DynamoDB.
        return calendar.timegm(value.timetuple())


class DynamoDbPySparkBinaryTypeConverter(DynamoDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_dynamodb(value: bytearray) -> bytearray:
        """
        No converter is needed since PySpark python native representation for BinaryType is same
        as DynamoDB.
        """
        return value


class DynamoDbPySparkDecimalTypeConverter(DynamoDbPySparkTypeConverter):
    """
    DynamoDB converter for Spark's Decimal data types. No converter is needed since PySpark python
    native representation for DecimalType is same as DynamoDB. DynamoDB supports 38 digits of
    precision which is equal to Pyspark Decimal precision.
    """

    @staticmethod
    @return_if_none
    def to_dynamodb(value: Decimal) -> Decimal:
        return value


# Complex data type converters.
# Complex data type converters are stateful and stores the type converter for underlying data.


class DynamoDbPySparkArrayTypeConverter(DynamoDbPySparkTypeConverter):
    def __init__(self, element_converter) -> None:
        self._element_converter = element_converter

    @return_if_none
    def to_dynamodb(self, value: list) -> list:
        return [self._element_converter.to_dynamodb(x) for x in value]


class DynamoDbPySparkMapTypeConverter(DynamoDbPySparkTypeConverter):
    def __init__(self, key_converter, value_converter) -> None:
        self._key_converter = key_converter
        self._value_converter = value_converter

    @return_if_none
    def to_dynamodb(self, value: dict) -> dict:
        # Dynamo Map type requires string key.
        return {
            str(self._key_converter.to_dynamodb(k)): self._value_converter.to_dynamodb(
                v
            )
            for k, v in value.items()
        }


# Helpers to get the appropriate data type converter

BASIC_DATA_TYPE_CONVERTERS = {
    BooleanType: DynamoDbPySparkBooleanTypeConverter,
    TimestampType: DynamoDbPySparkTimestampTypeConverter,
    DateType: DynamoDbPySparkDateTypeConverter,
    StringType: DynamoDbPySparkStringTypeConverter,
    FloatType: DynamoDbPySparkFloatDoubleTypeConverter,
    DoubleType: DynamoDbPySparkFloatDoubleTypeConverter,
    ShortType: DynamoDbPySparkShortIntLongTypeConverter,
    IntegerType: DynamoDbPySparkShortIntLongTypeConverter,
    LongType: DynamoDbPySparkShortIntLongTypeConverter,
    BinaryType: DynamoDbPySparkBinaryTypeConverter,
    DecimalType: DynamoDbPySparkDecimalTypeConverter,
}

COMPLEX_DATA_TYPE_CONVERTERS = {
    ArrayType: DynamoDbPySparkArrayTypeConverter,
    MapType: DynamoDbPySparkMapTypeConverter,
}


def get_type_converter(data_type: DataType) -> DynamoDbPySparkTypeConverter:
    """Helper utility that maps pyspark types to DynamoDB converter classes."""
    high_level_data_type = type(data_type)
    if high_level_data_type in BASIC_DATA_TYPE_CONVERTERS:
        return BASIC_DATA_TYPE_CONVERTERS[high_level_data_type]
    elif high_level_data_type in COMPLEX_DATA_TYPE_CONVERTERS:
        if high_level_data_type == ArrayType:
            return COMPLEX_DATA_TYPE_CONVERTERS[high_level_data_type](
                get_type_converter(data_type.elementType)
            )
        elif high_level_data_type == MapType:
            return COMPLEX_DATA_TYPE_CONVERTERS[high_level_data_type](
                get_type_converter(data_type.keyType),
                get_type_converter(data_type.valueType),
            )
    raise ValueError(f"Unsupported data type for DynamoDB publish: {data_type}")


def get_timestamp_key_type(data_type):
    if isinstance(data_type, StringType):
        return DYNAMODB_STRING_TYPE
    elif (
        isinstance(data_type, IntegerType)
        or isinstance(data_type, FloatType)
        or isinstance(data_type, DoubleType)
        or isinstance(data_type, LongType)
        or isinstance(data_type, TimestampType)
        or isinstance(data_type, DateType)
    ):
        return DYNAMODB_NUMBER_TYPE

    # Unsupported types for timestamp_keys: boolean, Struct, Map
    raise TypeError("Unsupported data type.")
