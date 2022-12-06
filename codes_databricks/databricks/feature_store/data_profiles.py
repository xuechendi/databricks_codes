"""Module for configure and compute the data profiles for feature tables."""

import functools
from typing import Iterable, List, Optional

from pyspark.sql import Column, DataFrame, functions as F, types as T
from databricks.feature_store.entities.data_profiles_entities import (
    Granularities,
    ComputationMode,
)
from databricks.feature_store import FeatureStoreClient

_GRANULARITY_COLUMN = "granularity"
_WINDOW_COLUMN = "window"


class DataProfilesSpec:
    def __init__(
        self,
        granularities: Iterable[str] = ("GLOBAL",),
        computation_mode: str = "MANUAL",
        features: Optional[Iterable[str]] = None,
    ):
        """
        This class serves as a configuration of the data profiles.

        :param granularities: Defines the window size when aggregating data based on the table's `timestamp_keys`.
          Currently only the following granularities are supported: {“GLOBAL“, “1 HOUR”, “1 DAY”, “1 WEEK”, “1 MONTH“}.
          “GLOBAL“ means the whole table will be grouped together.
        :param computation_mode: controls when a profile is re-computed. Valid values:
          - "MANUAL": No automatic triggering, user has to invoke manually by calling `refresh_data_profiles` or
                      through the UI.
          - "AUTO": Profiles are re-computed automatically when the feature table is updated. Streaming update is not
                    supported on this mode.
        :param features: A list of table column names to be computed for the data profiles. If this field is None,
          then all the columns will be included.
        """
        supported_computation_modes = [c.value for c in ComputationMode]
        if computation_mode.upper() not in supported_computation_modes:
            raise ValueError(
                f"Unsupported computation_mode: '{computation_mode}'. Use one of {', '.join(supported_computation_modes)}."
            )

        supported_granularities = [g.value for g in Granularities]
        for granularity in granularities:
            if granularity.upper() not in supported_granularities:
                raise ValueError(
                    f"Unsupported granularities: '{granularity}'. Use one of {', '.join(supported_granularities)}."
                )

        self._granularities = [
            Granularities(granularity.upper()) for granularity in granularities
        ]
        self._computation_mode = ComputationMode(computation_mode.upper())
        self._features = features

    @property
    def granularities(self) -> Iterable[str]:
        return [granularity.value for granularity in self._granularities]

    @property
    def computation_mode(self) -> str:
        return self._computation_mode.value

    @property
    def features(self):
        return self._features


def _generate_window_col_exp(timestamp_col: str, granularity: str) -> Column:
    """
    Generates the window column that will be used for aggregation.

    :param timestamp_col: The name of the timestamp column
    :param granularity: The granularity of the time window grouping
    :return: A struct columns with the nested fields 'start' and 'end'
    """
    if granularity == Granularities.GLOBAL:
        # Use NULL to represent window start/end for global aggregation
        return F.struct(F.lit(None).alias("start"), F.lit(None).alias("end"))
    elif granularity == Granularities.ONE_MONTH:
        # "1 month" is not explicitly supported by F.window, so we need to construct
        # the output manually with a struct.
        start_date = F.date_trunc("month", F.col(timestamp_col)).alias("start")
        end_date = F.date_trunc("month", F.add_months(start_date, 1)).alias("end")
        return F.struct(start_date, end_date)
    else:
        # If the granularity is 1 week, then we need to shift the window calculation by 4 days,
        # since the start of the epoch falls on Th.
        # This shift makes week windows start on Monday and end on Sunday.
        start_time = "4 days" if granularity == Granularities.ONE_WEEK else None
        # Note that this function assumes a compatible granularity string, see
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.window.html
        # Spark natively supports "1 hour", "1 day", and "1 week".
        return F.window(
            F.col(timestamp_col),
            windowDuration=granularity,
            slideDuration=granularity,
            startTime=start_time,
        )


def _compute_data_profiles(
    df: DataFrame, spec: DataProfilesSpec, ts_keys: List[str]
) -> DataFrame:
    """
    This internal function computes the data profiles given the DataFrame.

    Uses the provided timestamp keys and profile spec to compute a profile
    DataFrame for every granularity, and unions the resulting DataFrames
    into a single output.

    :param df: The data to compute the profiles.
    :param spec: The configuration that defines how profiles should be computed.
    :param ts_keys: The timestamp keys associated with the table.
    :return: The computed data profiles in a DataFrame with columns:
                - granularity
                - window
                - column_name
                - <all computed statistics>
    """
    if not ts_keys and spec.granularities != ["GLOBAL"]:
        raise ValueError(
            "Only 'GLOBAL' granularity is supported for tables without timestamp keys."
        )

    # Feature tables can have at most one timestamp key
    ts_key = ts_keys[0] if ts_keys else None

    features_to_compute = spec.features if spec.features else df.columns

    # TODO(ML-21787): Refactor so that we can reuse the metrics definition here.
    # Generates a struct that contains an expression for every aggregate metric we want to compute.
    aggregates_expressions = [
        F.struct(
            [
                F.count(column_name).alias("count"),
                F.sum(F.col(column_name).isNull().cast(T.LongType())).alias(
                    "num_nulls"
                ),
            ]
        ).alias(column_name)
        for column_name in features_to_compute
    ]
    # For each feature, we map the column name to itself and explode it so we can compute
    # the same set of aggregate expressions on every input feature
    map_args_for_explode = []
    for column_name in features_to_compute:
        map_args_for_explode.extend([F.lit(column_name), F.col(column_name)])
    transposed_columns = F.explode(F.create_map(*map_args_for_explode))

    # We will compute an aggregate DataFrame for every granularity from the spec and union them at the end
    aggregate_dfs = []
    for granularity in spec.granularities:
        # Create the granularity and window column to define this aggregation grouping,
        # and preserve these meta columns in the output schema.
        granularity_col = F.lit(granularity).alias(_GRANULARITY_COLUMN)
        window_col = _generate_window_col_exp(ts_key, granularity).alias(_WINDOW_COLUMN)
        meta_columns = [granularity_col, window_col]
        meta_column_names = [F.col(_GRANULARITY_COLUMN), F.col(_WINDOW_COLUMN)]

        aggregate_dfs.append(
            df.groupBy(meta_columns)
            .agg(*aggregates_expressions)
            .select(meta_column_names + [transposed_columns])
            .select(
                meta_column_names
                + [
                    F.col("key").alias("column_name"),
                    F.col("value.count"),
                    F.col("value.num_nulls"),
                ]
            )
        )

    return functools.reduce(DataFrame.unionByName, aggregate_dfs)


def configure_data_profiles(table_name: str, data_profiles_spec: DataProfilesSpec):
    """
    Overwrites the configuration of the data profiles of the feature table. Only those with `Can Manage` permission
    on the table can call this function.

    :param table_name: A feature table name of the form <database_name>.<table_name>, for example dev.user_features.
    :param data_profiles_spec: An instance of `DataProfilesSpec` which defined the behavior of the data profiles
      computation.
    """
    raise NotImplementedError()


def compute_data_profiles(
    table_name: str,
    granularities: Iterable[str] = ("GLOBAL",),
    features: Optional[Iterable[str]] = None,
):
    """
    Compute and return data profiles of the table. This will not overwrite the data profiles saved on the feature
    table. Only those with `Can View Metadata` permission on the table can call this function.

    :param table_name: A feature table name of the form <database_name>.<table_name>, for example dev.user_features.
    :param granularities: List of granularities to use when aggregating data into time windows based on
      the table's `timestamp_keys`.
      Currently only the following granularities are supported: {“GLOBAL“, “1 HOUR”, “1 DAY”, “1 WEEK”, “1 MONTH“}.
      If None, then the whole table will be grouped together.
    :param features: A list of table column names to be computed for the data profiles. If this field is None,
          then all the columns will be included.
    """
    fs = FeatureStoreClient()
    df = fs.read_table(table_name)
    ts_keys = fs.get_table(table_name).timestamp_keys
    spec = DataProfilesSpec(granularities=granularities, features=features)
    return _compute_data_profiles(df, spec, ts_keys)


def refresh_data_profiles(table_name: str):
    """
    Refresh the data profiles of the table. It re-computes the data profiles and saves the result. The data profiles
    must have been enabled before calling this API. Only the one with `Can Edit Metadata` permission on the table can
    call this function.

    :param table_name: A feature table name of the form <database_name>.<table_name>, for example dev.user_features.
    """
    raise NotImplementedError()
