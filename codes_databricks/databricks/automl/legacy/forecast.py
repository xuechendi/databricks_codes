import itertools
import logging
import re
import time
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

from databricks.automl_runtime.forecast import utils, OFFSET_ALIAS_MAP
import mlflow.entities
import numpy as np
import pandas as pd
import pyspark.pandas as ps
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, StructType, StructField, StringType
from pyspark.sql.window import Window

from databricks.automl.legacy.alerts import DatasetTruncatedAlert, MissingTimeStepsInTimeSeriesAlert, \
    ExecutionTimeoutAlert, ExtraTimeStepsInTimeSeriesAlert, UnmatchedFrequencyInTimeSeriesAlert, \
    CreateSchemaNotPermittedAlert, CreateTableNotPermittedAlert, TruncateHorizonAlert
from databricks.automl.legacy.alerts.alert_manager import AlertManager
from databricks.automl.legacy.base_learner import BaseLearner
from databricks.automl.legacy.confs import ForecastConf, UnivariateForecastConf
from databricks.automl.legacy.const import DatasetFormat, Framework, TimeSeriesFrequency
from databricks.automl.legacy.context import Context, DataSource, NotebookDetails
from databricks.automl.legacy.errors import ExecutionTimeoutError, ExecutionResultError, InvalidArgumentError
from databricks.automl.legacy.forecast_planner import ProphetPlanner, ArimaPlanner
from databricks.automl.legacy.forecast_preprocess import ForecastDataPreprocessor, ForecastDataPreprocessResults
from databricks.automl.legacy.imputers import Imputer
from databricks.automl.legacy.instrumentation import instrumented, log_forecast_data_stats
from databricks.automl.legacy.plan import Plan
from databricks.automl.legacy.planner import TrialPlanner
from databricks.automl.legacy.problem_type import ProblemType, Metric
from databricks.automl.legacy.result import AutoMLSummary, TrialInfo
from databricks.automl.legacy.sections.exploration.data import InputDataExplorationForecasting
from databricks.automl.legacy.stats import StatsCalculator

_logger = logging.getLogger(__name__)


class Forecast(BaseLearner):
    """
    Implementation of databricks.automl.forecast().
    """

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.FORECAST

    @property
    def default_metric(self) -> Metric:
        return Metric.SMAPE

    @classmethod
    def supported_metrics(cls) -> Iterable[Metric]:
        # Metric.MAPE and Metric.COVERAGE are removed due to issues caused by prophet removing mape/coverage
        # in `performance_metrics` under some cases. See details in ML-16769.
        # TODO: ML-16903 Handle the missing metrics in forecasting.
        return [Metric.MSE, Metric.RMSE, Metric.MAE, Metric.MDAPE, Metric.SMAPE]

    @property
    def supported_frameworks(self) -> List[Framework]:
        return [Framework.PROPHET, Framework.ARIMA]

    @classmethod
    def _get_supported_target_types(cls) -> List[DataType]:
        return ForecastDataPreprocessor.NUMERIC_TYPES

    @classmethod
    def _get_supported_time_types(cls) -> List[DataType]:
        return ForecastDataPreprocessor.TIME_TYPES

    def _get_planners(self, is_ds_frequency_consistent: bool,
                      exclude_frameworks: Set[Framework]) -> List[Type[TrialPlanner]]:
        planners = [ArimaPlanner, ProphetPlanner
                    ] if is_ds_frequency_consistent else [ProphetPlanner]
        if exclude_frameworks:
            planners = [
                planner for planner in planners if planner.framework() not in exclude_frameworks
            ]
        if len(planners) == 0:
            # This happens only if ARIMA is removed by AutoML and other frameworks are excluded by user. If all
            # frameworks are excluded by user, an exception will be thrown earlier in _validate_exclude_frameworks.
            raise InvalidArgumentError(
                "No supported frameworks left. The time series and specified frequency doesn't work for ARIMA "
                "model, while all other frameworks are excluded.")
        return planners

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
        frequency_in_seconds = TimeSeriesFrequency._member_map_[frequency].amount_in_seconds
        inconsistent_count = diff.filter((F.col("diff") % frequency_in_seconds) != 0).count()
        return inconsistent_count == 0

    @staticmethod
    def _is_ds_uniformly_spaced(dataset: DataFrame, time_col: str,
                                identity_col: Optional[List[str]], frequency: str,
                                is_ds_frequency_consistent: bool) -> bool:
        """
        Check if the time series is uniformly spaced given that the result from the _is_ds_frequency_consistent check.
        """
        frequency_in_seconds = TimeSeriesFrequency._member_map_[frequency].amount_in_seconds
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

    def _validate_params(
            self,
            timeout_minutes: Optional[int],
            horizon: Optional[int] = None,
            frequency: Optional[str] = None,
    ) -> None:
        super()._validate_params(timeout_minutes, horizon, frequency)
        if horizon <= 0:
            raise InvalidArgumentError("Forecast horizon must be greater than 0.")

        valid_frequency = TimeSeriesFrequency._member_names_
        if frequency not in valid_frequency:
            raise InvalidArgumentError(f"Forecast frequency should be one of {str(valid_frequency)}")

    def _validate_or_create_output_database(self, output_database: Optional[str],
                                            alert_manager: AlertManager) -> bool:
        if output_database is None:
            return False
        try:
            schema_exist = self.spark._jsparkSession.catalog().databaseExists(output_database)
            if not schema_exist:
                _logger.warning(f"Schema {output_database} does not exist.")
                self.spark.sql(f"CREATE SCHEMA {output_database}")
            return True
        except Exception:
            alert_manager.record(CreateSchemaNotPermittedAlert())
            return False

    @staticmethod
    def _truncate_time_series(dataset: DataFrame, time_col: str, horizon: int, frequency: str,
                              initial: int, alert_manager: AlertManager) -> DataFrame:
        """
        Truncate the time series. Only keep the data in the latest (num_folds+initial)* horizon period time.
        """
        # The time range for truncation should be smaller than pd.Timedelta.max which is the upper bound
        # for pd.Timedelta.
        max_horizon = pd.Timedelta.max / pd.Timedelta(1, frequency)
        time_delta = pd.Timedelta(value=initial*horizon, unit=frequency) \
            if initial*horizon < max_horizon else pd.Timedelta.max

        time_bound = dataset.select([F.max(time_col)]).collect()[0][0] - time_delta
        dataset_truncate = dataset.filter((F.col(time_col) > time_bound))

        if dataset.count() != dataset_truncate.count():
            print(
                f"Warning: AutoML will truncate the data by ignoring the data before {time_bound}.")
            alert_manager.record(DatasetTruncatedAlert())
        return dataset_truncate

    @staticmethod
    def _record_horizon_alert(dataset: DataFrame, horizon: int, frequency: str, target_col: str,
                              time_col: str, identity_col: Optional[List[str]],
                              alert_manager: AlertManager):
        frequency_unit = OFFSET_ALIAS_MAP[frequency]

        if not identity_col:
            df_agg = dataset.groupby(time_col).agg(
                F.avg(target_col).alias(f"avg_{target_col}")).withColumnRenamed(time_col, "ds")
            validation_horizon = utils.get_validation_horizon(df_agg.toPandas(), horizon,
                                                              frequency_unit)

            if validation_horizon < horizon:
                alert_manager.record(TruncateHorizonAlert([]))
        else:
            group_cols = [time_col] + identity_col
            df_agg = dataset.groupby(group_cols).agg(
                F.avg(target_col).alias(f"avg_{target_col}")).withColumnRenamed(
                    time_col, "ds").withColumn("ts_id", F.concat_ws('-', *identity_col))

            def get_horizon(df):
                validation_horizon = utils.get_validation_horizon(df, horizon, frequency_unit)
                ts_id = str(df["ts_id"].iloc[0])
                return pd.DataFrame([[validation_horizon, ts_id]], columns=["val_horizon", "ts_id"])

            truncated = df_agg.groupby(identity_col).applyInPandas(
                get_horizon,
                "val_horizon long, ts_id string").filter(F.col("val_horizon") < horizon)
            truncated_ts_ids = [row.ts_id for row in truncated.select("ts_id").collect()]

            if len(truncated_ts_ids) > 0:
                alert_manager.record(TruncateHorizonAlert(truncated_ts_ids))

    @instrumented
    def fit(self,
            dataset: Union[DataFrame, pd.DataFrame, ps.DataFrame],
            *,
            target_col: str,
            time_col: str,
            identity_col: Optional[Union[str, list]] = None,
            horizon: int = 1,
            frequency: str = "d",
            exclude_frameworks: List[str] = None,
            data_dir: Optional[str] = None,
            metric: Optional[Union[Metric, str]] = None,
            max_trials: int = UnivariateForecastConf().max_trials,
            parallelism: int = BaseLearner.MAX_TRIAL_PARALLELISM,
            timeout_minutes: Optional[int] = None,
            output_database: Optional[str] = None,
            experiment: Optional[mlflow.entities.Experiment] = None,
            home_dir: Optional[str] = None,
            run_id: Optional[str] = None) -> AutoMLSummary:
        """
        :param dataset: input Spark or pandas DataFrame that contains training features and targets.
        :param target_col: column name of the target labels.
        :param time_col: Optional column name of the time column for forecasting.
                         If not provided, we will detect the timestamp/data col as tome_col.
        :param identity_col: Optional list of column names of the identity columns for multi-series forecasting.
        :param horizon: the length of time into the future for which forecasts are to be prepared.
        :param frequency: Optional frequency of the time series. This represents the period with which events are expected to occur.
                        The frequency must be a pandas offset alias.
        :param exclude_frameworks: Frameworks that will be ignored by AutoML. The supported frameworks are "prophet"
                                   and "arima".
        :param data_dir: Optional DBFS path that is visible to both driver and worker nodes used to store
                         intermediate data
        :param metric: The metric that will be optimized across trials. Specified either as a concrete Metric or
                       the corresponding short hand. If None then it is set to default_metric.
        :param max_trials: The maximum number of trials to run. When timeout=None, maximum number of trials will run
                        to completion.
        :param parallelism: The maximum parallelism to use when running trials.
                        The actual parallelism is subject to available Spark task slots at
                        runtime.
                        If set to None (default) or a non-positive value, this will be set to
                        Spark's default parallelism or `1`.
                        We cap the value at `MAX_CONCURRENT_JOBS_ALLOWED=128`.
        :param timeout_minutes: The maximum time to wait for the AutoML trials to complete. timeout_minutes=None
                                will run the trials without any timeout restrictions.
        :param output_database: Schema name to save the predicted data. AutoML will create a new table in the schema
                            with the predicted data. If it is None, AutoML will not save any results.
        :param experiment: MLflow experiment to log this AutoML run to. This experiment should be new and unused.
                           If no experiment is given, a brand new one will be created.
        :param run_id: UUID set by @instrumented decorator. Do not set this manually.
        :param home_dir: This is the same as experiment_dir but with a different name to keep backwards compatibility
                         with the usage of this internal API by our AutoML UI:
                         https://github.com/databricks/universe/blob/master/webapp/web/js/mlflow/autoML/AutoMLDriverNotebook.ipynb

        :return: Structured summary object with info about trials.
        """
        # First available log that indicates the start of an AutoML run
        _logger.info(f"AutoML run for {self.problem_type.value} started with run_id: {run_id}")
        _logger.debug(
            f"AutoML called with params: target_col={target_col}, time_col={time_col} "
            f"identity_col={identity_col} horizon={horizon} frequency={frequency} "
            f"data_dir={data_dir} exclude_frameworks={exclude_frameworks} "
            f"metric={metric} max_trials={max_trials} timeout_minutes={timeout_minutes} "
            f"experiment_id={experiment.experiment_id if experiment is not None else None} "
            f"experiment_dir={home_dir}")

        return self._fit_run(
            dataset=dataset,
            target_col=target_col,
            data_dir=data_dir,
            exclude_cols=[],
            exclude_frameworks=exclude_frameworks,
            imputers={},
            metric=metric,
            max_trials=max_trials,
            parallelism=parallelism,
            timeout_minutes=timeout_minutes,
            experiment=experiment,
            experiment_dir=home_dir,
            time_col=time_col,
            identity_col=identity_col,
            horizon=horizon,
            frequency=frequency,
            output_database=output_database,
            run_id=run_id)

    def _fit_impl(
            self,
            dataset: DataFrame,
            target_col: str,
            context: Context,
            data_dir: Optional[str],
            metric: Metric,
            dataset_format: DatasetFormat,
            max_trials: int,
            parallelism: int,
            timeout: Optional[int],
            time_col: Optional[str],
            alert_manager: AlertManager,
            run_id: str,
            exclude_frameworks: Set[Framework],
            imputers: Dict[str, Imputer],  # supervised_learner only. Not used for forecasting
            num_exclude_cols: int,
            identity_col: Optional[Union[str, list]] = None,  # forecasting only
            horizon: Optional[int] = None,  # forecasting only
            frequency: Optional[str] = None,  # forecasting only
            output_database: Optional[str] = None,  # forecasting only
            pos_label: Optional[Any] = None,  # binary classification only
    ) -> AutoMLSummary:
        # extract identity_column and convert it to a list
        if identity_col is not None and type(identity_col) == str:
            identity_col = [identity_col]

        notebook_job_info = []

        should_save_result = self._validate_or_create_output_database(
            output_database, alert_manager=alert_manager)

        # start recording wall clock time
        start_time = datetime.now()

        cluster_info = self._validate_cluster()

        pre_sampling_stats = StatsCalculator.get_pre_sampling_stats(
            dataset=dataset,
            target_col=target_col,
            problem_type=self.problem_type,
        )

        _logger.debug(f"DataStats calculated before sampling: {pre_sampling_stats}")

        dataset = self._drop_invalid_rows(
            dataset=dataset,
            target_col=target_col,
            dataset_stats=pre_sampling_stats,
            alert_manager=alert_manager,
        )

        confs = ForecastConf.get_conf(identity_col)

        data_preprocessor = ForecastDataPreprocessor(
            pre_sampling_stats=pre_sampling_stats,
            dataset_schema=dataset.schema,
            target_col=target_col,
            time_col=time_col,
            identity_col=identity_col,
            horizon=horizon,
            supported_target_types=self._get_supported_target_types(),
            supported_time_types=self._get_supported_time_types(),
            alert_manager=alert_manager,
            confs=confs)

        dataset = self._truncate_time_series(dataset, time_col, horizon, frequency,
                                             confs.initial_period, alert_manager)

        self._record_horizon_alert(dataset, horizon, frequency, target_col, time_col, identity_col,
                                   alert_manager)

        data_source, dataset = context.save_training_data(
            dataset=dataset,
            data_dir=data_dir,
            data_format=DatasetFormat.SPARK,
            selected_cols=data_preprocessor.selected_cols,
        )

        _logger.debug(f"Saved the dataset to {data_source}")

        # This allows the user to check progress on MLflow while jobs are still running
        # TODO ML-12660: Convert this to a clickable URL
        experiment_url = context.get_experiment_url(sort_metric=metric)

        data_exp_nb_details, data_exp_result = self._run_data_exploration(
            context=context,
            data_source=data_source,
            target_col=target_col,
            time_col=time_col,
            identity_col=identity_col,
            timeout=timeout,
            experiment_url=experiment_url,
            cluster_info=cluster_info,
            alerts=alert_manager.get_displayable_alerts(),
        )
        notebook_job_info.append({
            "path": data_exp_nb_details.path,
            "run_id": data_exp_result.get("workflow_run_id")
        })

        _logger.info("Data exploration complete. Notebook can be found at: "
                     f"{data_exp_nb_details.url}")
        context.set_experiment_exploration_notebook(data_exp_nb_details.id)

        preprocess_result = data_preprocessor.process_data_exploration_result(data_exp_result)

        _logger.debug(f"Data Processing Results: {preprocess_result}")
        log_forecast_data_stats(
            spark=self.spark,
            notebook_id=data_exp_nb_details.id,
            num_target_nulls=pre_sampling_stats.num_target_nulls,
            num_rows=pre_sampling_stats.num_rows,
            num_cols=sum([len(cols) for cols in pre_sampling_stats.schema_map.values()]),
            run_id=run_id,
        )

        print(
            self._get_post_cmd_instructions(
                experiment_url=context.get_experiment_url(sort_metric=metric, absolute=True),
                metric=metric))

        # Construct the hyper-parameter search space for the trials
        is_ds_frequency_consistent = self._is_ds_frequency_consistent(dataset, time_col,
                                                                      identity_col, frequency)
        is_ds_uniformly_spaced = self._is_ds_uniformly_spaced(dataset, time_col, identity_col,
                                                              frequency, is_ds_frequency_consistent)
        if not is_ds_frequency_consistent:
            if is_ds_uniformly_spaced:
                alert_manager.record(UnmatchedFrequencyInTimeSeriesAlert(time_col))
            else:
                alert_manager.record(ExtraTimeStepsInTimeSeriesAlert(time_col))
        elif not is_ds_uniformly_spaced:
            alert_manager.record(MissingTimeStepsInTimeSeriesAlert(time_col))
        planners = self._get_planners(is_ds_frequency_consistent, exclude_frameworks)
        search_space = self._get_search_space(planners, max_trials)

        # subtract elapsed time for data exploration and setup from timeout to pass to hyperopt
        training_start_time = datetime.now()
        if timeout:
            data_exploration_time = int((training_start_time - start_time).total_seconds())
            timeout_current = timeout - data_exploration_time
            if timeout_current <= 0:
                alert_manager.record(ExecutionTimeoutAlert())
                raise ExecutionTimeoutError(
                    "Execution timed out before any trials could begin. "
                    "Please increase the timeout in order to allow trials to complete.")

        timedout_pattern = re.compile(".*TIMEDOUT.*")

        def _run_trial(params) -> Union[TrialInfo, Exception]:
            training_start_time = datetime.now()
            timeout_current = None
            if timeout:
                spend_time = int((training_start_time - start_time).total_seconds())
                timeout_current = timeout - spend_time
                if timeout_current < 0:
                    return ExecutionTimeoutError("Already timed out, the trial is not started")
            planner_random_state = params.pop("random_state")
            try:
                return self._run_trial_notebooks(
                    context=context,
                    data_source=data_source,
                    target_col=target_col,
                    time_col=time_col,
                    identity_col=identity_col,
                    horizon=horizon,
                    frequency=frequency,
                    preprocess_result=preprocess_result,
                    metric=metric,
                    max_evals=confs.max_evals,
                    timeout=timeout_current,
                    experiment_url=experiment_url,
                    cluster_info=cluster_info,
                    planner_random_state=planner_random_state,
                    **params)
            except Exception as e:
                if timedout_pattern.search(str(e)) is not None:
                    return ExecutionTimeoutError(str(e))
                else:
                    # This is a non-timeout error so let's just log the error
                    _logger.warning(f"Trial failed with {repr(e)}")
                    return e

        # Run the trials in parallel.
        pool = ThreadPool(confs.num_trials_in_parallel)
        # We set the random_state in the main thread (outside of _run_trial) to make sure each trial gets a
        # deterministic seed independent to the order of execution.
        for params in search_space:
            params["random_state"] = np.random.randint(1e9)
        trial_or_error = pool.map(_run_trial, search_space)
        trial_infos = [t for t in trial_or_error if not isinstance(t, Exception)]
        notebook_job_info.extend([{
            "path": t.notebook_path,
            "run_id": t.workflow_run_id
        } for t in trial_infos])

        # If at least one trial timed out, we raise ExecutionTimeoutError, otherwise ExecutionResultError.
        if len(trial_infos) == 0:
            num_timeout_trials = len(
                [e for e in trial_or_error if isinstance(e, ExecutionTimeoutError)])
            if num_timeout_trials == len(trial_or_error):
                raise ExecutionTimeoutError(
                    "All trials timed out. Please increase the timeout in order to allow trials to complete."
                )
            elif num_timeout_trials > 0:
                raise ExecutionTimeoutError(
                    "Some trials timed out while others failed."
                    "Please increase the timeout in order to allow trials to complete.")
            else:
                raise ExecutionResultError("All trials failed.")

        # sort trials by val metric, so best trial is first
        trials = self._sort_trials(trial_infos, metric)
        best_trial = trials[0]

        # TODO: We should check the graph output and retry the notebook export if it does not exist.
        time.sleep(30)
        context.save_job_notebooks(notebook_job_info)

        output_table_name = None
        if should_save_result:
            output_table_name = f"{output_database}.forecast_prediction_{context._session_id}"
            try:
                self._save_result_data(best_trial, dataset, target_col, time_col, output_table_name,
                                       identity_col)
                context.set_output_table_name(output_table_name)
            except Exception as e:
                _logger.warning(f"Failed to save the forecast data: {e}")
                alert_manager.record(CreateTableNotPermittedAlert())

        context.display_html(
            self._get_summary_html(
                trial=best_trial,
                data_exp_url=data_exp_nb_details.url,
                experiment_url=experiment_url,
                metric=metric))
        context.set_experiment_best_trial_notebook(best_trial.notebook_id)

        return AutoMLSummary(
            preprocess_result=preprocess_result,
            experiment=context.get_experiment(),
            trials=trials,
            output_table_name=output_table_name)

    @staticmethod
    def _run_trial_notebooks(context: Context, data_source: DataSource, target_col: str,
                             time_col: str, identity_col: Optional[List[str]], horizon: int,
                             frequency: str, preprocess_result: ForecastDataPreprocessResults,
                             metric: Metric, max_evals: int, timeout: Optional[int],
                             experiment_url: str, cluster_info: Dict[str, str],
                             planner_random_state: int, **kwargs) -> TrialInfo:
        var_target_col = "target_col"
        var_time_col = "time_col"
        var_id_cols = "id_cols"
        var_horizon = "horizon"
        var_frequency_unit = "unit"

        planner = kwargs["model"]
        trial_plan = planner(
            var_target_col=var_target_col,
            var_time_col=var_time_col,
            var_id_cols=var_id_cols,
            var_horizon=var_horizon,
            var_frequency_unit=var_frequency_unit,
            data_source=data_source,
            preprocess_result=preprocess_result,
            target_col=target_col,
            time_col=time_col,
            identity_col=identity_col,
            horizon=horizon,
            unit=frequency,
            metric=metric.short_name,
            max_evals=max_evals,
            timeout=timeout,
            experiment_id=context.experiment_id,
            experiment_url=experiment_url,
            driver_notebook_url=context.driver_notebook_url,
            cluster_info=cluster_info,
            random_state=planner_random_state,
            **kwargs).generate()

        trial = context.execute_trial_job(
            plan=trial_plan, metric=metric, flavor=planner.mlflow_flavor(), timeout=timeout)

        return trial

    @staticmethod
    def _run_data_exploration(
            context: Context,
            data_source: DataSource,
            target_col: str,
            time_col: str,
            identity_col: Optional[List[str]],
            timeout: Optional[int],
            experiment_url: str,
            cluster_info: Dict[str, str],
            alerts: Dict[str, List[str]],
    ) -> Tuple[NotebookDetails, Dict[str, Any]]:
        # generate and run the data exploration notebook
        data_exploration_plan = Plan(
            name="DataExploration",
            sections=[
                InputDataExplorationForecasting(
                    data_source=data_source,
                    target_col=target_col,
                    time_col=time_col,
                    identity_col=identity_col,
                    experiment_url=experiment_url,
                    driver_notebook_url=context.driver_notebook_url,
                    cluster_info=cluster_info,
                    alerts=alerts,
                )
            ])
        try:
            return context.execute_data_exploration_job(
                plan=data_exploration_plan, data_source=data_source, timeout=timeout)
        except ExecutionTimeoutError as e:
            raise ExecutionTimeoutError(
                "Execution timed out during data exploration. " +
                "Please increase the timeout in order to finish data exploration and run trials."
            ) from e

    @staticmethod
    def _get_search_space(planners: List[Type[TrialPlanner]],
                          max_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        search_space = []
        for planner in planners:
            model_search_spaces = planner.get_hyperparameter_search_space()
            space_tuples = [[(name, value) for value in values]
                            for name, values in model_search_spaces.items()]
            model_search_spaces = list(itertools.product(*space_tuples))

            search_space += [dict(item) for item in model_search_spaces]

        num_trial_notebooks = len(search_space)
        if max_trials and num_trial_notebooks > max_trials:
            search_space = search_space[:max_trials]
        return search_space

    @classmethod
    def _get_metrics_to_display(cls, metric: Metric,
                                metrics: Dict[str, float]) -> List[Tuple[str, Tuple[str, str]]]:
        """
        Returns a dictionary with the key as metric name (without prefix) and value as a tuple of
        strings with (validation, train) metric values for the metric name.

        We also round the metric to 3 decimal points and return None if any of the metrics aren't logged

        :param metrics: metric dictionary logged into MLflow
        :return: {metric_name: (val_metric, train_metrics)}
        """
        val_metrics = {}

        for metric_name, value in metrics.items():
            val_metrics[metric_name.replace("val_", "")] = "{:.3f}".format(value)

        display_metrics = {}
        for metric_name in set(val_metrics.keys()):
            val = val_metrics.get(metric_name, "None")
            display_metrics[metric_name] = val

        # re-arrange the metrics to display the primary metric at the top
        metric = metric.trial_metric_name.replace("val_", "")
        if metric in display_metrics.keys():
            primary_metric = (metric, display_metrics[metric])
            del display_metrics[metric]

            display_metrics = [(k, v) for k, v in display_metrics.items()]
            return [primary_metric] + display_metrics
        return [(k, v) for k, v in display_metrics.items()]

    @classmethod
    def _get_metric_table(cls, metric: Metric, trial: TrialInfo) -> str:
        formatted_rows = [
            f"""
            <tr>
                <th> {metric_name} </th>
                <td> {tst} </td>
            </tr>
            """ for metric_name, tst in cls._get_metrics_to_display(metric, trial.metrics)
        ]
        rows = "\n".join(formatted_rows)

        return f"""
                <table class="dataframe">
                    <thead>
                      <tr>
                        <th></th>
                        <th>Validation</th>
                      </tr>
                    </thead>
                    <tbody>
                    {rows}
                    </tbody>
                </table>
        """

    def _sort_trials(self, trials: List[TrialInfo], metric: Metric) -> List[TrialInfo]:
        """
        Special sort function to handle trials that have trained partial models
        :param trials: TrialInfo of completed trials
        :param metric: Metric to sort by
        :return:
        """
        # try using trials that don't have partial model
        complete_trials = [
            trial for trial in trials if not bool(trial.params.get("partial_model", False))
        ]
        partial_trials = [
            trial for trial in trials if bool(trial.params.get("partial_model", False))
        ]

        if len(complete_trials) == 0:
            _logger.warning("All trials have trained partial models")
            complete_trials = trials
            partial_trials = []

        complete_trials = super()._sort_trials(complete_trials, metric)
        return complete_trials + partial_trials

    def _save_result_data(self,
                          best_trial: TrialInfo,
                          dataset: DataFrame,
                          target_col: str,
                          time_col: str,
                          output_table_name: str,
                          identity_col: Optional[list] = None) -> None:
        trial_id = best_trial.mlflow_run_id
        model_uri = f"runs:/{trial_id}/model"
        forecast_model = mlflow.pyfunc.load_model(model_uri)
        forecast_pdf = forecast_model._model_impl.python_model.predict_timeseries(
            include_history=False)
        # Save table for univariate forecasting
        if identity_col is None:
            output_database = StructType([
                StructField(time_col, dataset.schema[time_col].dataType, True),
                StructField(target_col, dataset.schema[target_col].dataType, True),
                StructField(f"{target_col}_lower", dataset.schema[target_col].dataType, True),
                StructField(f"{target_col}_upper", dataset.schema[target_col].dataType, True)
            ])
            selected_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        else:  # Save table for multi-series forecasting
            output_database = StructType([
                StructField("ts_id", StringType(), True),
                StructField(time_col, dataset.schema[time_col].dataType, True),
                StructField(target_col, dataset.schema[target_col].dataType, True),
                StructField(f"{target_col}_lower", dataset.schema[target_col].dataType, True),
                StructField(f"{target_col}_upper", dataset.schema[target_col].dataType, True)
            ])
            selected_cols = ["ts_id", "ds", "yhat", "yhat_lower", "yhat_upper"]
        forecast_df = self.spark.createDataFrame(
            forecast_pdf[selected_cols], schema=output_database)
        forecast_df.write.format("delta").saveAsTable(output_table_name)
