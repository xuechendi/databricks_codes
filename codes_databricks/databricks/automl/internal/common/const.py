from __future__ import annotations

from enum import Enum, unique

from pyspark.databricks.sql import annotation_utils
from pyspark.sql.types import ArrayType, BooleanType, ByteType, DateType, DoubleType, \
    FloatType, IntegerType, LongType, NumericType, ShortType, StringType, TimestampType, \
    DecimalType
from databricks.automl.shared.const import SemanticType


# https://docs.python.org/3/library/enum.html#ensuring-unique-enumeration-values
@unique
class ContextType(Enum):
    JUPYTER = 1
    DATABRICKS = 2


@unique
class DatasetFormat(Enum):
    SPARK = "spark"
    PANDAS = "pandas"
    PYSPARK_PANDAS = "pyspark.pandas"


@unique
class Framework(Enum):
    SKLEARN = "sklearn"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    PROPHET = "prophet"
    ARIMA = "arima"


class SparkDataType:
    INTEGER_TYPES = (
        ByteType,
        ShortType,
        IntegerType,
        LongType,
    )
    FLOATING_POINT_TYPES = (
        FloatType,
        DoubleType,
    )
    DECIMAL_TYPE = (DecimalType, )
    NUMERIC_TYPES = INTEGER_TYPES + FLOATING_POINT_TYPES + DECIMAL_TYPE
    TIME_TYPES = (TimestampType, DateType)
    STRING_TYPE = (StringType, )
    ALL_TYPES = NUMERIC_TYPES + TIME_TYPES + STRING_TYPE + (ArrayType, BooleanType)
    SPARK_TYPE_TO_PANDAS_TYPE = {"NumericType": "float", "TimestampType": "datetime"}

    UNSUPPORTED_TYPE = "unsupported"
    NUMERICAL_TYPE = "numeric"


@unique
class AutoMLDataType(Enum):
    """
    Similar to https://pandas-profiling.ydata.ai/docs/master/index.html#types
    """
    ARRAY = "Array"
    BOOLEAN = "Boolean"
    DATETIME = "DateTime"
    NUMERIC = "Numeric"
    STRING = "String"
    TEXT = "Text"

    @staticmethod
    def from_spark_type(spark_type):
        if isinstance(spark_type, BooleanType):
            return AutoMLDataType.BOOLEAN
        if isinstance(spark_type, NumericType):
            return AutoMLDataType.NUMERIC
        if isinstance(spark_type, DateType) or isinstance(spark_type, TimestampType):
            return AutoMLDataType.DATETIME
        if isinstance(spark_type, ArrayType):
            return AutoMLDataType.ARRAY
        if isinstance(spark_type, StringType):
            return AutoMLDataType.STRING
        raise TypeError(f"Invalid spark type {spark_type}")

    @staticmethod
    def from_semantic_type(semantic_type):
        if semantic_type == SemanticType.NUMERIC:
            return AutoMLDataType.NUMERIC
        if semantic_type == SemanticType.DATETIME:
            return AutoMLDataType.DATETIME
        if semantic_type == SemanticType.TEXT:
            return AutoMLDataType.TEXT
        if semantic_type == SemanticType.CATEGORICAL:
            return AutoMLDataType.STRING
        raise TypeError(f"Invalid semantic type {semantic_type}")


@unique
class RunState(Enum):
    PENDING = 1  # Currently does not appear in this package because it's only set from the UI
    RUNNING = 2
    SUCCESS = 3
    FAILED = 4
    CANCELED = 5  # @todo: ML-14217 Backend support for cancellation


@unique
class CloudProvider(Enum):
    """
    Cloud provider names corresponding to Spark conf for `spark.databricks.cloudProvider`.
    """
    AWS = "AWS"
    AZURE = "Azure"
    GCP = "GCP"


@unique
class DatabricksDocumentationDomain(Enum):
    """
    Documentation domain addresses for each Databricks cloud provider.
    """
    AWS = "https://docs.databricks.com"
    AZURE = "https://docs.microsoft.com/azure/databricks"
    GCP = "https://docs.gcp.databricks.com"


@unique
class SparseOrDense(Enum):
    SPARSE = "SPARSE"
    DENSE = "DENSE"
