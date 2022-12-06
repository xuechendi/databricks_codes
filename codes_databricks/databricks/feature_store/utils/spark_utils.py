""" Spark utility function. """

from pyspark.sql.types import ShortType
from pyspark.sql.functions import to_json
from databricks.feature_store.constants import COMPLEX_DATA_TYPES
from databricks.feature_store.entities.data_type import DataType


def _get_columns_of_spark_type(schema, spark_type):
    return [
        field.name for field in schema.fields if isinstance(field.dataType, spark_type)
    ]


def get_columns_of_type_short(schema):
    return _get_columns_of_spark_type(schema, ShortType)


def serialize_complex_data_types(df):
    """
    Serialize all columns in df that are of complex datatype using Spark's to_json() udf.
    """
    schema = df.schema
    select_column_list = [
        to_json(column).alias(column)
        if DataType.from_spark_type(schema[column].dataType) in COMPLEX_DATA_TYPES
        else column
        for column in df.columns
    ]
    return df.select(select_column_list)
