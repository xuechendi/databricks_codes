from databricks.feature_store.entities.feature_spec import FeatureSpec
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from typing import Set, List
import copy
import re

SINGLE_LEVEL_NAMESPACE_TABLE_REGEX = r"^[\w_]+$"
TWO_LEVEL_NAMESPACE_TABLE_REGEX = r"^[\w_]+(\.[\w_]+)$"
THREE_LEVEL_NAMESPACE_TABLE_REGEX = r"^[\w_]+(\.[\w_]+){2}$"

HIVE_METASTORE_NAME = "hive_metastore"
# these two catalog names both points to the workspace local default HMS (hive metastore).
LOCAL_METASTORE_NAMES = [HIVE_METASTORE_NAME, "spark_catalog"]
# samples catalog is managed by databricks for hosting public dataset like NYC taxi dataset.
# it is neither a UC nor local metastore catalog, accessing samples catalog using feature store client is not allowed.
SAMPLES_CATALOG_NAME = "samples"


# Get full table name in the form of <catalog_name>.<database_name>.<table_name>
# given user specified table name, current catalog and schema.
def get_full_table_name(
    table_name: str,
    current_catalog: str,
    current_schema: str,
) -> str:
    if not _is_single_level_name(current_catalog) or not _is_single_level_name(
        current_schema
    ):
        raise ValueError(
            f"Invalid catalog '{current_catalog}' or "
            f"schema '{current_schema}' name for table '{table_name}'."
        )
    _check_qualified_names({table_name})
    if _is_single_level_name(table_name):
        full_table_name = f"{current_catalog}.{current_schema}.{table_name}"
    elif _is_two_level_name(table_name):
        full_table_name = f"{current_catalog}.{table_name}"
    elif _is_three_level_name(table_name):
        full_table_name = table_name
    else:
        raise _invalid_table_names_error({table_name})
    catalog, schema, table = full_table_name.split(".")
    if catalog == SAMPLES_CATALOG_NAME:
        raise ValueError(
            "'samples' catalog cannot be accessed using feature store client."
        )
    if catalog in LOCAL_METASTORE_NAMES:
        return f"{HIVE_METASTORE_NAME}.{schema}.{table}"
    return full_table_name


# Get full table name for tables in feature lookups and reformat table names
def get_feature_lookups_with_full_table_names(
    feature_lookups: List[FeatureLookup], current_catalog: str, current_schema: str
) -> List[FeatureLookup]:
    table_names = {fl.table_name for fl in feature_lookups}
    _check_qualified_names(table_names)
    _verify_all_tables_are_either_in_uc_or_in_hms(
        table_names, current_catalog, current_schema
    )
    standardized_feature_lookups = []
    for fl in feature_lookups:
        fl_copy = copy.deepcopy(fl)
        fl_copy._table_name = get_full_table_name(
            fl_copy.table_name, current_catalog, current_schema
        )
        standardized_feature_lookups.append(fl_copy)
    return standardized_feature_lookups


# Local metastore tables in feature_spec.yaml are all stored in 2L.
# Standardize table names to be all in 3L to avoid erroneously reading data from UC tables.
def get_feature_spec_with_full_table_names(feature_spec: FeatureSpec) -> FeatureSpec:
    column_info_table_names = [
        column_info.table_name for column_info in feature_spec.feature_column_infos
    ]
    table_info_table_names = [
        table_info.table_name for table_info in feature_spec.table_infos
    ]
    _check_qualified_names(set(column_info_table_names))
    _check_qualified_names(set(table_info_table_names))
    invalid_table_names = list(
        filter(_is_single_level_name, column_info_table_names)
    ) + list(filter(_is_single_level_name, table_info_table_names))
    if len(invalid_table_names) > 0:
        raise _invalid_table_names_error(set(invalid_table_names))
    standardized_feature_spec = copy.deepcopy(feature_spec)
    for column_info in standardized_feature_spec.feature_column_infos:
        if _is_two_level_name(column_info.table_name):
            column_info._table_name = f"{HIVE_METASTORE_NAME}.{column_info.table_name}"
    for table_info in standardized_feature_spec.table_infos:
        if _is_two_level_name(table_info.table_name):
            table_info._table_name = f"{HIVE_METASTORE_NAME}.{table_info.table_name}"
    return standardized_feature_spec


# Reformat 3L table name for tables in local metastore to 2L. This is used when interacting with catalog client
# and serializing workspace local feature spec for scoring.
def reformat_full_table_name(full_table_name: str) -> str:
    if not _is_three_level_name(full_table_name):
        raise _invalid_table_names_error({full_table_name})
    catalog, schema, table = full_table_name.split(".")
    if catalog in LOCAL_METASTORE_NAMES:
        return f"{schema}.{table}"
    return full_table_name


# Reformat table names in feature_spec with reformat_full_table_name
def get_feature_spec_with_reformat_full_table_names(
    feature_spec: FeatureSpec,
) -> FeatureSpec:
    column_info_table_names = [
        column_info.table_name for column_info in feature_spec.feature_column_infos
    ]
    table_info_table_names = [
        table_info.table_name for table_info in feature_spec.table_infos
    ]
    _check_qualified_names(set(column_info_table_names))
    _check_qualified_names(set(table_info_table_names))
    invalid_table_names = list(
        filter(lambda name: not _is_three_level_name(name), column_info_table_names)
    ) + list(
        filter(lambda name: not _is_three_level_name(name), table_info_table_names)
    )
    if len(invalid_table_names) > 0:
        raise _invalid_table_names_error(set(invalid_table_names))
    standardized_feature_spec = copy.deepcopy(feature_spec)
    for column_info in standardized_feature_spec.feature_column_infos:
        column_info._table_name = reformat_full_table_name(column_info.table_name)
    for table_info in standardized_feature_spec.table_infos:
        table_info._table_name = reformat_full_table_name(table_info.table_name)
    return standardized_feature_spec


def _invalid_table_names_error(invalid_table_names: Set[str]) -> ValueError:
    return ValueError(
        f"Invalid table name{'s' if len(invalid_table_names) > 1 else ''} '{', '.join(invalid_table_names)}'."
    )


def _is_qualified_table_name(feature_table_name) -> bool:
    return isinstance(feature_table_name, str) and (
        _is_single_level_name(feature_table_name)
        or _is_two_level_name(feature_table_name)
        or _is_three_level_name(feature_table_name)
    )


def _is_single_level_name(name) -> bool:
    return (
        isinstance(name, str)
        and re.match(SINGLE_LEVEL_NAMESPACE_TABLE_REGEX, name) is not None
    )


def _is_two_level_name(name) -> bool:
    return (
        isinstance(name, str)
        and re.match(TWO_LEVEL_NAMESPACE_TABLE_REGEX, name) is not None
    )


def _is_three_level_name(name) -> bool:
    return (
        isinstance(name, str)
        and re.match(THREE_LEVEL_NAMESPACE_TABLE_REGEX, name) is not None
    )


# check if table is in UC
def _is_uc_table(table_name: str, current_catalog: str, current_schema: str) -> bool:
    full_table_name = get_full_table_name(table_name, current_catalog, current_schema)
    catalog_name, schema_name, table_name = full_table_name.split(".")
    return (
        not _is_default_hms_table(full_table_name, current_catalog, current_schema)
        and catalog_name != SAMPLES_CATALOG_NAME
    )


# check if table is in default HMS
def _is_default_hms_table(
    table_name: str, current_catalog: str, current_schema: str
) -> bool:
    full_table_name = get_full_table_name(table_name, current_catalog, current_schema)
    catalog_name, schema_name, table_name = full_table_name.split(".")
    return (
        catalog_name in LOCAL_METASTORE_NAMES and catalog_name != SAMPLES_CATALOG_NAME
    )


# check if table names are in the correct format - 1L, 2L or 3L
def _check_qualified_names(feature_table_names: Set[str]):
    unqualified_table_names = list(
        filter(
            lambda table_name: not _is_qualified_table_name(table_name),
            feature_table_names,
        )
    )
    if len(unqualified_table_names) > 0:
        raise ValueError(
            f"Feature table name{'s' if len(unqualified_table_names) > 1 else ''} "
            f"'{', '.join(map(str, unqualified_table_names))}' must have the form "
            f"<catalog_name>.<schema_name>.<table_name>, <database_name>.<table_name>, "
            f"or <table_name> and only contain alphabet characters, numbers and _."
        )


# For APIs like create_training_set and score_batch, all tables must all be in
# UC catalog (shareable cross-workspaces) or default HMS (intended to only be used in the current workspace)
# check if all tables are either in UC or default HMS.
def _verify_all_tables_are_either_in_uc_or_in_hms(
    table_names: Set[str], current_catalog: str, current_schema: str
):
    is_valid = all(
        map(
            lambda table_name: _is_uc_table(
                table_name, current_catalog, current_schema
            ),
            table_names,
        )
    ) or all(
        map(
            lambda table_name: _is_default_hms_table(
                table_name, current_catalog, current_schema
            ),
            table_names,
        )
    )
    if not is_valid:
        raise ValueError(
            f"Feature table names '{', '.join(table_names)}' "
            f"must all be in UC or the local default hive metastore. "
            f"Mixing feature tables from two different storage locations is not allowed."
        )
