import logging
from dataclasses import dataclass
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from databricks.automl.shared.const import TimeSeriesFrequency
from databricks.automl.internal.alerts import ExtraTimeStepsInTimeSeriesAlert, \
    MissingTimeStepsInTimeSeriesAlert, NotEnoughHistoricalDataAlert, \
    TimeSeriesIdentitiesTooShortAlert, UnmatchedFrequencyInTimeSeriesAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.shared.errors import UnsupportedDataError

_logger = logging.getLogger(__name__)


@dataclass
class ForecastPostSamplingStats:
    valid_time_series: List[str]
    invalid_identities: List[str]
    is_ds_frequency_consistent: bool
    is_ds_uniformly_spaced: bool


class ForecastPostSamplingStatsCalculator:
    MIN_TIME_SERIES_LENGTH = 5

    @staticmethod
    def get_post_sampling_stats(dataset: DataFrame, target_col: str, time_col: str,
                                identity_col: Optional[List[str]],
                                frequency: str) -> ForecastPostSamplingStats:
        time_group = [time_col] + identity_col if identity_col else [time_col]
        df_aggregated = dataset.groupby(time_group) \
            .agg(F.avg(target_col).alias(target_col))
        if identity_col:
            time_series_count = df_aggregated.select([target_col] + identity_col) \
                .groupby(identity_col).count() \
                .select([F.concat_ws("-", *identity_col).alias("ts_id"), "count"]) \
                .toPandas().set_index("ts_id")["count"].to_dict()
        else:
            time_series_count = {"total_num": df_aggregated.select([target_col]).count()}

        valid_time_series = [
            k for k, v in time_series_count.items()
            if v >= ForecastPostSamplingStatsCalculator.MIN_TIME_SERIES_LENGTH
        ]
        invalid_identities = [
            k for k, v in time_series_count.items()
            if v < ForecastPostSamplingStatsCalculator.MIN_TIME_SERIES_LENGTH
        ]

        is_ds_frequency_consistent = ForecastPostSamplingStatsCalculator. \
            _is_ds_frequency_consistent(dataset, time_col,
                                        identity_col, frequency)
        is_ds_uniformly_spaced = ForecastPostSamplingStatsCalculator._is_ds_uniformly_spaced(
            dataset, time_col, identity_col, frequency, is_ds_frequency_consistent)

        return ForecastPostSamplingStats(
            valid_time_series=valid_time_series,
            invalid_identities=invalid_identities,
            is_ds_frequency_consistent=is_ds_frequency_consistent,
            is_ds_uniformly_spaced=is_ds_uniformly_spaced,
        )

    @staticmethod
    def validate_post_sampling_stats(
            time_col: str,
            post_sampling_stats: ForecastPostSamplingStats,
            alert_manager: AlertManager,
    ) -> None:
        valid_time_series = post_sampling_stats.valid_time_series
        invalid_identities = post_sampling_stats.invalid_identities
        if not valid_time_series:
            alert_manager.record(NotEnoughHistoricalDataAlert())
            raise UnsupportedDataError(f"Not enough time series data for training and validation. "
                                       f"Please provide longer time series data.")
        elif len(invalid_identities) > 0:
            alert_manager.record(TimeSeriesIdentitiesTooShortAlert(invalid_identities))
            _logger.warning(
                f"Time series with the following identities are too short: {invalid_identities}. "
                f"Models won't be trained for these identities. "
                f"Please provide longer time series data.")

        is_ds_frequency_consistent = post_sampling_stats.is_ds_frequency_consistent
        is_ds_uniformly_spaced = post_sampling_stats.is_ds_uniformly_spaced
        if not is_ds_frequency_consistent:
            if is_ds_uniformly_spaced:
                alert_manager.record(UnmatchedFrequencyInTimeSeriesAlert(time_col))
            else:
                alert_manager.record(ExtraTimeStepsInTimeSeriesAlert(time_col))
        elif not is_ds_uniformly_spaced:
            alert_manager.record(MissingTimeStepsInTimeSeriesAlert(time_col))

    @staticmethod
    def _is_ds_frequency_consistent(dataset: DataFrame, time_col: str,
                                    identity_col: Optional[List[str]], frequency: str) -> bool:
        """
        Check whether the time series frequency is consistent with given frequency unit so that it works for
        the ARIMA trial. Consistency here means that the time series only contains timestamps in the format of
        start_ds + k * frequency, where k is int values.
        """
        if identity_col:
            epoch_seconds = dataset.select(
                F.unix_timestamp(time_col).cast("long").alias(time_col), *identity_col)
            window_var = Window.partitionBy(*identity_col)
            diff = epoch_seconds.select(
                (F.col(time_col) - F.min(time_col).over(window_var)).alias("diff"))
        else:
            epoch_seconds = dataset.select(F.unix_timestamp(time_col).cast("long").alias(time_col))
            start_ds = epoch_seconds.agg(F.min(time_col)).collect()[0][0]
            diff = epoch_seconds.select((F.col(time_col) - start_ds).alias("diff"))
        frequency_in_seconds = TimeSeriesFrequency._member_map_[frequency].value_in_seconds
        inconsistent_count = diff.filter((F.col("diff") % frequency_in_seconds) != 0).count()
        return inconsistent_count == 0

    @staticmethod
    def _is_ds_uniformly_spaced(dataset: DataFrame, time_col: str,
                                identity_col: Optional[List[str]], frequency: str,
                                is_ds_frequency_consistent: bool) -> bool:
        """
        Check if the time series is uniformly spaced given that the result from the _is_ds_frequency_consistent check.
        """
        frequency_in_seconds = TimeSeriesFrequency._member_map_[frequency].value_in_seconds
        if identity_col:
            df_stats = dataset.groupBy(*identity_col). \
                agg(F.min(time_col).alias("start_ds"),
                    F.max(time_col).alias("end_ds"),
                    F.count(time_col).alias("count"))
            df_stats = df_stats.withColumn(
                "duration_epoch_seconds",
                F.unix_timestamp("end_ds").cast("long") - F.unix_timestamp("start_ds").cast("long"))
            if is_ds_frequency_consistent:
                # In this case, the only possibility of the time series being not uniformly spaced is that there
                # are missing time steps.
                df_expected_count = df_stats.withColumn(
                    "expected_count", 1 + F.col("duration_epoch_seconds") / frequency_in_seconds)
                missing_count = df_expected_count.filter(
                    F.col("count") != F.col("expected_count")).count()
                return missing_count == 0
            else:
                # In this case, we check if each time series itself is uniformly spaced and if the frequencies are
                # the same for different time series
                df_stats = df_stats.withColumn(
                    "possible_frequency_in_seconds",
                    F.col("duration_epoch_seconds") / (F.col("count") - 1))
                non_integer_count = df_stats.filter(
                    (F.col("possible_frequency_in_seconds") % 1) != 0).count()
                if non_integer_count != 0:
                    return False
                possible_frequency_in_seconds = df_stats.select(
                    "possible_frequency_in_seconds").first()[0]
                different_frequency_count = df_stats.filter(
                    F.col("possible_frequency_in_seconds") != possible_frequency_in_seconds).count(
                    )
                if different_frequency_count != 0:
                    return False
                epoch_seconds = dataset.select(
                    F.unix_timestamp(time_col).cast("long").alias(time_col), *identity_col)
                window_var = Window().partitionBy(*identity_col)
                diff = epoch_seconds.select(
                    (F.col(time_col) - F.min(time_col).over(window_var)).alias("diff"),
                    *identity_col)
                diff = diff.join(
                    df_stats.select("possible_frequency_in_seconds", *identity_col),
                    on=identity_col)
                non_divisible_count = diff.filter(
                    (F.col("diff") % F.col("possible_frequency_in_seconds")) != 0).count()
                return non_divisible_count == 0
        else:
            epoch_seconds = dataset.select(F.unix_timestamp(time_col).cast("long").alias(time_col))
            start_ds, end_ds = epoch_seconds.agg(F.min(time_col), F.max(time_col)).collect()[0]
            if is_ds_frequency_consistent:
                # In this case, the only possibility of the time series being not uniformly spaced is that there
                # are missing time steps
                expected_count = 1 + (end_ds - start_ds) / frequency_in_seconds
                return expected_count == dataset.count()
            else:
                # In this case, we check if the time series itself is uniformly spaced or not
                possible_frequency_in_seconds = (end_ds - start_ds) / (dataset.count() - 1)
                if not float.is_integer(possible_frequency_in_seconds):
                    return False
                diff = epoch_seconds.select((F.col(time_col) - start_ds).alias("diff"))
                non_divisible_count = diff.filter(
                    (F.col("diff") % possible_frequency_in_seconds) != 0).count()
                return non_divisible_count == 0
