import logging

from databricks.feature_store.constants import DATA_TYPES_REQUIRES_DETAILS
from databricks.feature_store.entities.data_type import DataType
from databricks.feature_store.constants import _WARN, _ERROR

_logger = logging.getLogger(__name__)


def catalog_matches_delta_schema(catalog_features, df_schema, column_filter=None):
    """
    Confirm that the column names and column types are the same.

    Returns True if identical, False if there is a mismatch.

    If column_filter is not None, only columns in column_filter must match.
    """
    if column_filter is not None:
        catalog_features = [c for c in catalog_features if c.name in column_filter]
        df_schema = [c for c in df_schema if c.name in column_filter]

    catalog_schema = {
        feature.name: DataType.from_string(feature.data_type)
        for feature in catalog_features
    }
    delta_schema = {
        feature.name: DataType.from_spark_type(feature.dataType)
        for feature in df_schema
    }

    complex_catalog_schema = get_complex_catalog_schema(
        catalog_features, catalog_schema
    )
    complex_delta_schema = get_complex_delta_schema(df_schema, delta_schema)

    return (
        catalog_schema == delta_schema
        and complex_catalog_schema == complex_delta_schema
    )


def get_complex_delta_schema(delta_features, delta_feature_names_to_fs_types):
    """
    1. Filter delta features to features that have complex datatypes.
    2. Take the existing Spark DataType stored on the Delta features. This is later used for
    comparison against the Catalog schema's complex Spark DataTypes.
    3. Return a mapping of feature name to their respective complex Spark DataTypes.

    :param delta_features: List[Feature]. List of features stored in Delta.
    :param delta_feature_names_to_fs_types: Map[str, feature_store.DataType]. A mapping of feature
    names to their respective Feature Store DataTypes.
    :return: Map[str, spark.sql.types.DataType]. A mapping of feature names to their respective
    Spark DataTypes.
    """
    complex_delta_features = [
        feature
        for feature in delta_features
        if delta_feature_names_to_fs_types[feature.name] in DATA_TYPES_REQUIRES_DETAILS
    ]
    complex_delta_feature_names_to_spark_types = {
        feature.name: feature.dataType for feature in complex_delta_features
    }
    return complex_delta_feature_names_to_spark_types


def get_complex_catalog_schema(catalog_features, catalog_feature_names_to_fs_types):
    """
    1. Filter catalog features to features that have complex datatypes.
    2. Convert the JSON string stored in each feature's data_type_details to the corresponding
    Spark DataType. This is later used for comparison against the Delta schema's complex Spark
    DataTypes.
    3. Return a mapping of feature name to their respective complex Spark DataTypes.

    :param catalog_features: List[Feature]. List of features stored in the Catalog.
    :param catalog_feature_names_to_fs_types: Map[str, feature_store.DataType]. A mapping of feature
    names to their respective Feature Store DataTypes.
    :return: Map[str, spark.sql.types.DataType]. A mapping of feature names to their respective
    Spark DataTypes.
    """
    complex_catalog_features = [
        feature
        for feature in catalog_features
        if catalog_feature_names_to_fs_types[feature.name]
        in DATA_TYPES_REQUIRES_DETAILS
    ]
    complex_catalog_feature_names_to_spark_types = {
        feature.name: DataType.to_complex_spark_type(feature.data_type_details)
        for feature in complex_catalog_features
    }
    return complex_catalog_feature_names_to_spark_types


def log_catalog_schema_not_match_delta_schema(catalog_features, df_schema, level):
    """
    Log the catalog schema does not match the delta table schema.

    Example warning:
    Expected recorded schema from Feature Catalog to be identical with
    schema in delta table.Feature Catalog's schema is
    '{'id': 'INTEGER', 'feat1': 'INTEGER'}' while delta table's
    schema is '{'id': 'INTEGER', 'feat1': 'FLOAT'}'
    """
    catalog_schema = {feature.name: feature.data_type for feature in catalog_features}
    delta_schema = {
        feature.name: feature.dataType.typeName().upper() for feature in df_schema
    }
    msg = (
        f"Expected recorded schema from Feature Catalog to be identical with schema "
        f"in Delta table. "
        f"Feature Catalog's schema is '{catalog_schema}' while Delta table's schema "
        f"is '{delta_schema}'"
    )
    if level == _WARN:
        _logger.warning(msg)
    elif level == _ERROR:
        raise RuntimeError(msg)
    else:
        _logger.info(msg)
