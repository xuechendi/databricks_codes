from enum import auto, Enum, unique

from pyspark.databricks.sql import annotation_utils
from pyspark.sql.types import ArrayType, BooleanType, DateType, NumericType, StringType, TimestampType


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
class SemanticType(Enum):
    """
    Detected ML feature type of a column.
    """
    DATETIME = annotation_utils.DATETIME
    NUMERIC = annotation_utils.NUMERIC
    CATEGORICAL = annotation_utils.CATEGORICAL
    NATIVE = annotation_utils.NATIVE
    TEXT = annotation_utils.TEXT


@unique
class MLFlowFlavor(Enum):
    SKLEARN = 1
    PROPHET = 2
    ARIMA = 3


@unique
class RunState(Enum):
    PENDING = 1  # Currently does not appear in this package because it's only set from the UI
    RUNNING = 2
    SUCCESS = 3
    FAILED = 4
    CANCELED = 5  # @todo: ML-14217 Backend support for cancellation


@unique
class TimeSeriesFrequency(Enum):
    W = auto(), 604800
    d = auto(), 86400
    D = auto(), 86400
    days = auto(), 86400
    day = auto(), 86400
    hours = auto(), 3600
    hour = auto(), 3600
    hr = auto(), 3600
    h = auto(), 3600
    m = auto(), 60
    minute = auto(), 60
    min = auto(), 60
    minutes = auto(), 60
    T = auto(), 60
    S = auto(), 1
    seconds = auto(), 1
    sec = auto(), 1
    second = auto(), 1

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        obj._amount_in_seconds_ = args[1]
        return obj

    @property
    def amount_in_seconds(self):
        return self._amount_in_seconds_


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
