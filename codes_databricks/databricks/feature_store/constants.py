from databricks.feature_store.entities.data_type import DataType

OVERWRITE = "overwrite"
MERGE = "merge"
PATH = "path"
TABLE = "table"
CUSTOM = "custom"
PREDICTION_COLUMN_NAME = "prediction"
MODEL_DATA_PATH_ROOT = "feature_store"
UTF8_BYTES_PER_CHAR = 4
MAX_PRIMARY_KEY_STRING_LENGTH_CHARS = 100
MAX_PRIMARY_KEY_STRING_LENGTH_BYTES = (
    MAX_PRIMARY_KEY_STRING_LENGTH_CHARS * UTF8_BYTES_PER_CHAR
)
COMPLEX_DATA_TYPES = [DataType.ARRAY, DataType.MAP]
DATA_TYPES_REQUIRES_DETAILS = COMPLEX_DATA_TYPES + [DataType.DECIMAL]
_DEFAULT_WRITE_STREAM_TRIGGER = {"processingTime": "5 seconds"}
_DEFAULT_PUBLISH_STREAM_TRIGGER = {"processingTime": "5 minutes"}

_WARN = "WARN"
_ERROR = "ERROR"
_SOURCE_FORMAT_DELTA = "delta"