from typing import List, Dict
from collections import defaultdict

from databricks.feature_store.entities.data_type import DataType
from databricks.feature_store.entities.feature_column_info import FeatureColumnInfo
from databricks.feature_store.entities.source_data_column_info import (
    SourceDataColumnInfo,
)
from databricks.feature_store.entities.feature_spec import FeatureSpec
from databricks.feature_store.entities.feature_table import FeatureTable

from databricks.feature_store.utils import utils

from pyspark.sql import DataFrame


def _spark_asof_join_features(
    df: DataFrame,
    df_lookup_keys: List[str],
    df_timestamp_lookup_key: str,
    feature_table_data: DataFrame,
    feature_table_keys: List[str],
    feature_table_timestamp_key: str,
    feature_to_output_name: Dict[str, str],
) -> DataFrame:
    # [ML-19825] Import TSDF only at as-of join time to avoid DynamoDB publish issue
    from tempo.tsdf import TSDF

    # Alias feature table's keys to DataFrame lookup keys
    ft_key_aliases = [
        feature_table_data[ft_key].alias(df_key)
        for (ft_key, df_key) in zip(feature_table_keys, df_lookup_keys)
    ]
    # Alias features to corresponding output names
    ft_features = [
        (feature_name, output_name)
        for feature_name, output_name in feature_to_output_name.items()
        # Skip join if feature it is already in DataFrame and therefore overridden
        if output_name not in df.columns
    ]
    ft_feature_aliases = [
        feature_table_data[feature_name].alias(output_name)
        for feature_name, output_name in ft_features
    ]
    # Alias feature table's timestamp key to DataFrame timestamp lookup keys
    ft_timestamp_key_aliases = [
        feature_table_data[feature_table_timestamp_key].alias(df_timestamp_lookup_key)
    ]
    # Select key, timestamp key, and feature columns from feature table
    feature_and_keys = feature_table_data.select(
        ft_key_aliases + ft_timestamp_key_aliases + ft_feature_aliases
    )
    # Initialize Tempo TSDFs for df and feature table.
    # Feature table timestamp keys and primary keys are aliased to
    # DataFrame's timestamp lookup keys and lookup keys respectively
    # because ts_col and partition_cols must match.
    df_tsdf = TSDF(df, ts_col=df_timestamp_lookup_key, partition_cols=df_lookup_keys)
    ft_tsdf = TSDF(
        feature_and_keys,
        ts_col=df_timestamp_lookup_key,
        partition_cols=df_lookup_keys,
    )
    # Perform as-of join
    joined_df = df_tsdf.asofJoin(
        ft_tsdf, left_prefix="left", right_prefix="right", skipNulls=False
    ).df
    # Remove prefixes from joined df to restore column names
    left_aliases = [
        joined_df[f"left_{column_name}"].alias(column_name)
        for column_name in df.columns
        if column_name not in df_lookup_keys
    ]
    right_aliases = [
        joined_df[f"right_{output_name}"].alias(output_name)
        for (_, output_name) in ft_features
    ]
    return joined_df.select(df_lookup_keys + left_aliases + right_aliases)


def _spark_join_features(
    df: DataFrame,
    df_keys: List[str],
    feature_table_data: DataFrame,
    feature_table_keys: List[str],
    feature_to_output_name: Dict[str, str],
) -> DataFrame:
    """
    Helper to join `feature_name` from `feature_table_data` into `df`.

    This join uses a temporary table that contains only the keys and feature
    from the feature table. The temporary table aliases the keys to match
    the lookup keys and the feature to match the output_name.

    Aliasing the keys allows us to join on name instead of by column,
    which prevents duplicate column names after the join.
    (see: https://kb.databricks.com/data/join-two-dataframes-duplicated-columns.html)

    The joined-in feature is guaranteed to be unique because FeatureSpec
    columns must be unique and the join is skipped if the feature
    already exists in the DataFrame.
    """

    # Alias feature table's keys to DataFrame lookup keys
    ft_key_aliases = [
        feature_table_data[ft_key].alias(df_key)
        for (ft_key, df_key) in zip(feature_table_keys, df_keys)
    ]
    # Alias features to corresponding output names
    ft_feature_aliases = [
        feature_table_data[feature_name].alias(output_name)
        for feature_name, output_name in feature_to_output_name.items()
        # Skip join if feature it is already in DataFrame and therefore overridden
        if output_name not in df.columns
    ]
    # Select key and feature columns from feature table
    feature_and_keys = feature_table_data.select(ft_key_aliases + ft_feature_aliases)
    # Join feature to feature table
    return df.join(feature_and_keys, df_keys, how="left")


def _validate_join_keys(
    feature_column_info: FeatureColumnInfo,
    df: DataFrame,
    feature_table_metadata: FeatureTable,
    feature_table_data: DataFrame,
    is_timestamp_key: bool = False,
):
    join_error_phrase = (
        f"Unable to join feature table '{feature_column_info.table_name}'"
    )
    feature_column_info_keys = (
        feature_column_info.timestamp_lookup_key
        if is_timestamp_key
        else feature_column_info.lookup_key
    )
    feature_table_keys = (
        feature_table_metadata.timestamp_keys
        if is_timestamp_key
        else feature_table_metadata.primary_keys
    )

    lookup_key_kind = "timestamp lookup key" if is_timestamp_key else "lookup key"
    feature_table_key_kind = "timestamp key" if is_timestamp_key else "primary key"

    # Validate df has necessary keys
    missing_df_keys = list(
        filter(lambda df_key: df_key not in df.columns, feature_column_info_keys)
    )
    if missing_df_keys:
        missing_keys = ", ".join([f"'{key}'" for key in missing_df_keys])
        raise ValueError(
            f"{join_error_phrase} because {lookup_key_kind} {missing_keys} not found in DataFrame."
        )
    # Validate feature table has necessary keys
    missing_ft_keys = list(
        filter(
            lambda ft_key: ft_key not in feature_table_data.columns, feature_table_keys
        )
    )
    if missing_ft_keys:
        missing_keys = ", ".join([f"'{key}'" for key in missing_ft_keys])
        raise ValueError(
            f"{join_error_phrase} because {feature_table_key_kind} {missing_keys} not found in feature table."
        )

    # Validate number of feature table keys matches number of df lookup keys
    if len(feature_column_info_keys) != len(feature_table_keys):
        raise ValueError(
            f"{join_error_phrase} because "
            f"number of {feature_table_key_kind}s ({feature_table_keys}) "
            f"does not match "
            f"number of {lookup_key_kind}s ({feature_column_info_keys})."
        )

    # Validate feature table keys match types of df keys. The number of keys is expected to be the same.
    for (df_key, ft_key) in zip(feature_column_info_keys, feature_table_keys):
        df_key_type = DataType.from_spark_type(df.schema[df_key].dataType)
        ft_key_type = DataType.from_spark_type(
            feature_table_data.schema[ft_key].dataType
        )
        if df_key_type != ft_key_type:
            raise ValueError(
                f"{join_error_phrase} because {feature_table_key_kind} '{ft_key}' has type '{DataType.to_string(ft_key_type)}' "
                f"but corresponding {lookup_key_kind} '{df_key}' has type '{DataType.to_string(df_key_type)}' in DataFrame."
            )


def _validate_join_feature_data(
    feature_spec: FeatureSpec,
    df: DataFrame,
    extra_columns: List[str],
    feature_table_metadata_map: Dict[str, FeatureTable],
    feature_table_data_map: Dict[str, DataFrame],
):
    # Validate extra columns are unique
    if len(extra_columns) != len(set(extra_columns)):
        raise ValueError("extra_columns must be unique")
    # Validate df has extra columns
    for extra_column in extra_columns:
        if extra_column not in df.columns:
            raise ValueError(f"Column '{extra_column}' not found in DataFrame")
    for feature_info in feature_spec.feature_column_infos:
        feature_table_metadata = feature_table_metadata_map[feature_info.table_name]
        feature_table_data = feature_table_data_map[feature_info.table_name]
        # Validate feature table primary keys match length/type of df lookup keys
        _validate_join_keys(
            feature_info,
            df,
            feature_table_metadata,
            feature_table_data,
            is_timestamp_key=False,
        )
        # Validate feature table timestamp keys match length/type of df timestamp lookup keys
        _validate_join_keys(
            feature_info,
            df,
            feature_table_metadata,
            feature_table_data,
            is_timestamp_key=True,
        )


def join_feature_data(
    feature_spec: FeatureSpec,
    df: DataFrame,
    extra_columns: List[str],
    feature_table_metadata_map: Dict[str, FeatureTable],
    feature_table_data_map: Dict[str, DataFrame],
) -> DataFrame:
    """
    Joins `df` with features specified by `feature_spec`,
    using `feature_table_metadata_map` to determine join key
    and `feature_table_data_map` to retrieve the feature data.

    Before joining, it checks that
    1) `extra_columns` are in `df`
    2) `extra_columns` are unique
    3) feature table keys match length and types of `df` lookup keys specified by FeatureSpec
    4) `df` contains lookup keys specified by FeatureSpec
    5) feature table timestamp lookup keys match length and types of `df` timestamp lookup keys if specified by FeatureSpec
    6) `df` contains timestamp lookup keys if specified by FeatureSpec
    """
    _validate_join_feature_data(
        feature_spec=feature_spec,
        df=df,
        extra_columns=extra_columns,
        feature_table_metadata_map=feature_table_metadata_map,
        feature_table_data_map=feature_table_data_map,
    )

    # Helper class to group all unique combinations of feature table names and lookup keys.
    # All features in each of these groups will be JOINed with the training df using a single JOIN.
    class JoinDataKey:
        def __init__(
            self,
            feature_table: str,
            lookup_key: List[str],
            timestamp_lookup_key: List[str],
        ):
            self.feature_table = feature_table
            self.lookup_key = lookup_key
            self.timestamp_lookup_key = timestamp_lookup_key

        def __hash__(self):
            return (
                hash(self.feature_table)
                + hash(tuple(self.lookup_key))
                + hash(tuple(self.timestamp_lookup_key))
            )

        def __eq__(self, other):
            return (
                self.feature_table == other.feature_table
                and self.lookup_key == other.lookup_key
                and self.timestamp_lookup_key == other.timestamp_lookup_key
            )

    # Iterate through the list of FeatureColumnInfo and group features by name of the
    # feature table and lookup key(s) and timestamp lookup key(s)
    table_join_data = defaultdict(dict)
    for feature_info in feature_spec.feature_column_infos:
        join_data_key = JoinDataKey(
            feature_info.table_name,
            feature_info.lookup_key,
            feature_info.timestamp_lookup_key,
        )
        table_join_data[join_data_key][
            feature_info.feature_name
        ] = feature_info.output_name

    for join_data_key, feature_to_output_name in table_join_data.items():

        feature_table_metadata = feature_table_metadata_map[join_data_key.feature_table]
        feature_table_data = feature_table_data_map[join_data_key.feature_table]

        if join_data_key.timestamp_lookup_key:
            df = _spark_asof_join_features(
                df=df,
                df_lookup_keys=join_data_key.lookup_key,
                df_timestamp_lookup_key=join_data_key.timestamp_lookup_key[0],
                feature_table_data=feature_table_data,
                feature_table_keys=feature_table_metadata.primary_keys,
                feature_table_timestamp_key=feature_table_metadata.timestamp_keys[0],
                feature_to_output_name=feature_to_output_name,
            )
        else:
            df = _spark_join_features(
                df=df,
                df_keys=join_data_key.lookup_key,
                feature_table_data=feature_table_data,
                feature_table_keys=feature_table_metadata.primary_keys,
                feature_to_output_name=feature_to_output_name,
            )
    # Return the requested columns in the feature spec order with extra_columns appended at the end
    feature_spec_column_names = [info.output_name for info in feature_spec.column_infos]
    extra_column_names = [
        col for col in extra_columns if col not in feature_spec_column_names
    ]

    return df.select(feature_spec_column_names + extra_column_names)
