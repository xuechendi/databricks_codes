import base64
from abc import abstractmethod, ABC
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Type

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

# Abstract converters


class CosmosDbPySparkTypeConverter(ABC):
    @staticmethod
    @abstractmethod
    def to_cosmosdb(value):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def online_data_type() -> DataType:
        raise NotImplementedError


class CosmosDbPySparkMapKeyTypeConverter(CosmosDbPySparkTypeConverter):
    """
    Abstract class to handle conversion of Spark MapType key data types.
    Maps are stored as JSON documents in Cosmos DB, where the keys must be strings.
    """

    @staticmethod
    @abstractmethod
    def to_cosmosdb(value) -> str:
        raise NotImplementedError

    @staticmethod
    def online_data_type() -> DataType:
        return StringType()


# Converter implementations


class CosmosDbPySparkBinaryTypeConverter(CosmosDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_cosmosdb(value: bytearray) -> str:
        return str(base64.b64encode(value), "utf-8")

    @staticmethod
    def online_data_type() -> DataType:
        return StringType()


class CosmosDbPySparkDateTypeConverter(CosmosDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_cosmosdb(value: date) -> str:
        return value.isoformat()

    @staticmethod
    def online_data_type() -> DataType:
        return StringType()


class CosmosDbPySparkTimestampTypeConverter(CosmosDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_cosmosdb(value: datetime) -> str:
        # Spark Timestamps, represented as datetime in Python, are timezone naive and assumed to be in UTC.
        # We snap the timezone to UTC and then drop the timezone info. This ensures that timezone.isoformat()
        # for timezone naive (assumed UTC) and timezone aware datetimes will generate an naive UTC ISO string.
        # The expected format is yyyy-MM-ddTHH:mm:ss.ffffff without any timezone offset information.
        dt_naive_utc = value.astimezone(timezone.utc).replace(tzinfo=None)
        return dt_naive_utc.isoformat(timespec="microseconds")

    @staticmethod
    def online_data_type() -> DataType:
        return StringType()


class CosmosDbPySparkDecimalTypeConverter(CosmosDbPySparkTypeConverter):
    @staticmethod
    @return_if_none
    def to_cosmosdb(value: Decimal) -> str:
        return str(value)

    @staticmethod
    def online_data_type() -> DataType:
        return StringType()


class CosmosDbPySparkMapKeyStringTypeConverter(CosmosDbPySparkMapKeyTypeConverter):
    @staticmethod
    def to_cosmosdb(value: str) -> str:
        return value


class CosmosDbPySparkMapKeyShortIntLongTypeConverter(
    CosmosDbPySparkMapKeyTypeConverter
):
    @staticmethod
    def to_cosmosdb(value: int) -> str:
        return str(value)


# Identity, Array, and Map converters are stateful converters


class CosmosDbPySparkIdentityConverter(CosmosDbPySparkTypeConverter):
    """
    Helper identity converter for data types that do not require conversion.
    Data types that use this should explicitly initialize a dedicated converter,
    e.g. CosmosDbPySparkBooleanTypeConverter = CosmosDbPySparkIdentityConverter(BooleanType)
    """

    def __init__(self, dt_type: Type[DataType]):
        self._dt_type = dt_type

    def to_cosmosdb(self, value):
        return value

    def online_data_type(self) -> DataType:
        return self._dt_type()


class CosmosDbPySparkArrayTypeConverter(CosmosDbPySparkTypeConverter):
    def __init__(
        self, element_converter: CosmosDbPySparkTypeConverter, contains_null: bool
    ):
        self._element_converter = element_converter
        self._contains_null = contains_null

    @return_if_none
    def to_cosmosdb(self, value: list):
        return [self._element_converter.to_cosmosdb(e) for e in value]

    def online_data_type(self) -> DataType:
        return ArrayType(
            elementType=self._element_converter.online_data_type(),
            containsNull=self._contains_null,
        )


class CosmosDbPySparkMapTypeConverter(CosmosDbPySparkTypeConverter):
    def __init__(
        self,
        key_converter: CosmosDbPySparkMapKeyTypeConverter,
        value_converter: CosmosDbPySparkTypeConverter,
        value_contains_null: bool,
    ):
        self._key_converter = key_converter
        self._value_converter = value_converter
        self._value_contains_null = value_contains_null

    @return_if_none
    def to_cosmosdb(self, value: dict) -> dict:
        return {
            self._key_converter.to_cosmosdb(k): self._value_converter.to_cosmosdb(v)
            for k, v in value.items()
        }

    def online_data_type(self) -> DataType:
        return MapType(
            keyType=self._key_converter.online_data_type(),
            valueType=self._value_converter.online_data_type(),
            valueContainsNull=self._value_contains_null,
        )


CosmosDbPySparkBooleanTypeConverter = CosmosDbPySparkIdentityConverter(BooleanType)
CosmosDbPySparkShortTypeConverter = CosmosDbPySparkIdentityConverter(ShortType)
CosmosDbPySparkIntegerTypeConverter = CosmosDbPySparkIdentityConverter(IntegerType)
CosmosDbPySparkLongTypeConverter = CosmosDbPySparkIdentityConverter(LongType)
CosmosDbPySparkFloatTypeConverter = CosmosDbPySparkIdentityConverter(FloatType)
CosmosDbPySparkDoubleTypeConverter = CosmosDbPySparkIdentityConverter(DoubleType)
CosmosDbPySparkStringTypeConverter = CosmosDbPySparkIdentityConverter(StringType)

# Helpers to get the appropriate data type converter

BASIC_DATA_TYPE_IDENTITIES = {
    BooleanType: CosmosDbPySparkBooleanTypeConverter,
    ShortType: CosmosDbPySparkShortTypeConverter,
    IntegerType: CosmosDbPySparkIntegerTypeConverter,
    LongType: CosmosDbPySparkLongTypeConverter,
    FloatType: CosmosDbPySparkFloatTypeConverter,
    DoubleType: CosmosDbPySparkDoubleTypeConverter,
    StringType: CosmosDbPySparkStringTypeConverter,
}

BASIC_DATA_TYPE_CONVERTERS = {
    **BASIC_DATA_TYPE_IDENTITIES,
    BinaryType: CosmosDbPySparkBinaryTypeConverter,
    DateType: CosmosDbPySparkDateTypeConverter,
    TimestampType: CosmosDbPySparkTimestampTypeConverter,
    DecimalType: CosmosDbPySparkDecimalTypeConverter,
}

MAP_KEY_DATA_TYPE_CONVERTERS = {
    StringType: CosmosDbPySparkMapKeyStringTypeConverter,
    ShortType: CosmosDbPySparkMapKeyShortIntLongTypeConverter,
    IntegerType: CosmosDbPySparkMapKeyShortIntLongTypeConverter,
    LongType: CosmosDbPySparkMapKeyShortIntLongTypeConverter,
}


def get_data_type_converter(data_type: DataType) -> CosmosDbPySparkTypeConverter:
    dt_type = type(data_type)
    if dt_type == ArrayType:
        return CosmosDbPySparkArrayTypeConverter(
            element_converter=get_data_type_converter(data_type.elementType),
            contains_null=data_type.containsNull,
        )
    elif dt_type == MapType:
        return CosmosDbPySparkMapTypeConverter(
            key_converter=get_map_key_data_type_converter(data_type.keyType),
            value_converter=get_data_type_converter(data_type.valueType),
            value_contains_null=data_type.valueContainsNull,
        )
    else:
        return BASIC_DATA_TYPE_CONVERTERS[dt_type]


def get_map_key_data_type_converter(
    data_type: DataType,
) -> CosmosDbPySparkMapKeyTypeConverter:
    dt_type = type(data_type)
    if dt_type not in MAP_KEY_DATA_TYPE_CONVERTERS:
        raise ValueError(
            f"Unsupported map key data type for Cosmos DB publish: {dt_type}"
        )
    return MAP_KEY_DATA_TYPE_CONVERTERS[dt_type]
