import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import hyperopt
import mlflow.entities
import numpy as np
import pandas as pd
import pyspark.pandas as ps
from databricks.automl_runtime.hyperopt.early_stop import get_early_stop_fn
from hyperopt import fmin, STATUS_OK, SparkTrials
from hyperopt.spark import FMIN_CANCELLED_REASON_EARLY_STOPPING
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DataType, DateType, StringType

from databricks.automl.legacy.alerts import DatasetTooLargeAlert, EarlyStopAlert, ExecutionTimeoutAlert, \
    UnsupportedColumnAlert, UnsupportedTimeTypeAlert, DataExplorationFailAlert, NullsInTimeColumnAlert
from databricks.automl.legacy.alerts.alert_manager import AlertManager
from databricks.automl.legacy.base_learner import BaseLearner
from databricks.automl.legacy.confs import InternalConfs
from databricks.automl.legacy.const import ContextType, DatasetFormat, Framework, MLFlowFlavor, SemanticType
from databricks.automl.legacy.context import Context, DataSource, NotebookDetails
from databricks.automl.legacy.data_splitter import DataSplitter, RandomDataSplitter
from databricks.automl.legacy.errors import ExecutionTimeoutError, ExecutionResultError, \
    UnsupportedDataError, UnsupportedColumnError
from databricks.automl.legacy.imputers import Imputer
from databricks.automl.legacy.instrumentation import instrumented, log_supervised_data_stats, \
    log_supervised_input_data_stats
from databricks.automl.legacy.plan import Plan
from databricks.automl.legacy.preprocess import SupervisedLearnerDataPreprocessor, SupervisedLearnerDataPreprocessConfs, \
    SupervisedLearnerDataPreprocessResults
from databricks.automl.legacy.problem_type import ClassificationTargetTypes, Metric
from databricks.automl.legacy.result import AutoMLSummary, TrialInfo
from databricks.automl.legacy.sections.exploration.data import InputDataExplorationSupervisedLearner
from databricks.automl.legacy.size_estimator import SizeEstimator
from databricks.automl.legacy.stats import StatsCalculator, PreSamplingStats

_logger = logging.getLogger(__name__)

# ML-13588: Turn off the hyperopt warning. Remove after fixing hyperopt MLflow autologging issue.
_logger_hyperopt = logging.getLogger("hyperopt-spark")
_logger_hyperopt.setLevel(logging.ERROR)

# minimum number of trials to run before early stopping is considered
NO_EARLY_STOP_THRESHOLD = 200
# hyperopt search will stop early if the loss doesn't improve after this number of iterations
NO_PROGRESS_STOP_THRESHOLD = 100


class SupervisedLearner(BaseLearner):
    # Column name prefix of the sample weight column added by AutoML when balancing the dataset
    SAMPLE_WEIGHT_COL_PREFIX = "_automl_sample_weight"
    # When the dataset is imbalanced but not sampled because of memory estimation, downsample to this fraction
    BALANCED_SAMPLE_FRACTION = 0.7

    def __init__(
            self,
            context_type: ContextType,
            confs: SupervisedLearnerDataPreprocessConfs = SupervisedLearnerDataPreprocessor.CONFS):
        super().__init__(context_type)
        self._confs = confs

    @classmethod
    def _get_supported_time_types(cls) -> List[DataType]:
        return SupervisedLearnerDataPreprocessor.TIME_TYPES + \
               SupervisedLearnerDataPreprocessor.INTEGER_TYPES + \
               SupervisedLearnerDataPreprocessor.STRING_TYPE

    @property
    def splitter(self) -> DataSplitter:
        return RandomDataSplitter()

    @property
    def supported_frameworks(self) -> List[Framework]:
        return [Framework.SKLEARN, Framework.XGBOOST, Framework.LIGHTGBM]

    @instrumented
    def fit(
            self,
            dataset: Union[DataFrame, pd.DataFrame, ps.DataFrame],
            *,
            target_col: str,
            data_dir: Optional[str] = None,
            exclude_cols: List[str] = [],
            exclude_columns: List[str] = [],
            exclude_frameworks: List[str] = None,
            imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
            metric: Optional[Union[Metric, str]] = None,
            max_trials: Optional[
                int] = None,  # this is deprecated and only checked for deprecation warning
            parallelism: int = BaseLearner.MAX_TRIAL_PARALLELISM,
            timeout_minutes: Optional[int] = None,
            experiment: Optional[mlflow.entities.Experiment] = None,
            time_col: Optional[str] = None,
            run_id: Optional[str] = None,
            home_dir: Optional[str] = None,
            pos_label: Optional[ClassificationTargetTypes] = None) -> AutoMLSummary:
        """
        For user-facing parameters, which are not documented here, see docstrings in __init__.py
        :param parallelism: The maximum parallelism to use when running trials.
                        The actual parallelism is subject to available Spark task slots at
                        runtime.
                        If set to None (default) or a non-positive value, this will be set to
                        Spark's default parallelism or `1`.
                        We cap the value at `MAX_CONCURRENT_JOBS_ALLOWED=128`.
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
            f"AutoML called with params: target_col={target_col}, data_dir={data_dir} "
            f"exclude_cols={exclude_cols} exclude_columns={exclude_columns} exclude_frameworks={exclude_frameworks} "
            f"imputers={imputers} metric={metric} max_trials={max_trials} "
            f"timeout_minutes={timeout_minutes} "
            f"experiment_id={experiment.experiment_id if experiment is not None else None} "
            f"time_col={time_col} experiment_dir={home_dir} "
            f"pos_label={pos_label}")

        if max_trials != None:
            _logger.warning(
                "Parameter max_trials is deprecated and has no effect. The parameter will be removed in a future "
                "Databricks Runtime release. Chose timeout_minutes to control the duration of the AutoML runs. "
                "AutoML stops will automatically stop tuning models if the validation metric no longer improves."
            )
        if exclude_columns:
            _logger.warning(
                "Parameter exclude_columns is deprecated and will be removed in a future Databricks Runtime release. "
                "Please use exclude_cols instead.")
            if exclude_cols:
                _logger.warning(
                    "Both exclude_columns and exclude_cols are specified. The value of exclude_columns is ignored."
                )
            else:
                exclude_cols = exclude_columns

        return self._fit_run(
            dataset=dataset,
            target_col=target_col,
            data_dir=data_dir,
            exclude_cols=exclude_cols,
            exclude_frameworks=exclude_frameworks,
            imputers=imputers,
            metric=metric,
            max_trials=self.DEFAULT_MAX_TRIALS,
            parallelism=parallelism,
            timeout_minutes=timeout_minutes,
            experiment=experiment,
            experiment_dir=home_dir,
            time_col=time_col,
            run_id=run_id,
            pos_label=pos_label)

    @staticmethod
    def get_run_trial_fn(context: Context, metric: Metric, data_source: DataSource,
                         preprocess_result: SupervisedLearnerDataPreprocessResults, target_col: str,
                         time_col: Optional[str], imputers: Dict[str, Imputer], experiment_url: str,
                         cluster_info: Dict[str, str], sample_fraction: Optional[float],
                         planner_random_state: int, pos_label: Optional[ClassificationTargetTypes],
                         split_col: Optional[str], sample_weight_col: Optional[str]):
        """
        :param context: execution context
        :param metric: The metric that will be optimized across trials.
        :param data_source: DataSource
        :param preprocess_result: SupervisedLearnerDataPreprocessResults
        :param target_col: column name of the target labels.
        :param time_col: Optional column name of a time column. If provided, AutoML will try to split train/val/test
                         sets by time. Accepted column types are date/time, string and integer. If column type is
                         string AutoML will try to convert it to datetime by semantic detection, and the AutoML run
                         will fail if the conversion fails.
        :param experiment_url: URL of the MLflow experiment
        :param cluster_info: information about the cluster
        :param sample_fraction: Optional sampling fraction, used in trial notebook markdown cells
        :param planner_random_state: seed used in generated code sections
        :param pos_label: Optional positive class for binary classification.
        :param split_col: Optional name of the column that specifies the train/val/test split
        :param sample_weight_col: Optional name of the sample weight column
        :return: function used by hyperopt fmin to optimize hyperparameters by running training notebooks
        """

        def run_trial(params):
            hyperparameters = params.copy()
            trial_planner = hyperparameters.pop("model")

            trial_plan = trial_planner(
                var_target_col="target_col",
                var_time_col="time_col",
                var_X_train="X_train",
                var_X_val="X_val",
                var_X_test="X_test",
                var_y_train="y_train",
                var_y_val="y_val",
                var_y_test="y_test",
                var_preprocessor="preprocessor",
                var_run="mlflow_run",
                var_model="model",
                var_pipeline="pipeline",
                data_source=data_source,
                preprocess_result=preprocess_result,
                target_col=target_col,
                time_col=time_col,
                imputers=imputers,
                experiment_id=context.experiment_id,
                experiment_url=experiment_url,
                driver_notebook_url=context.driver_notebook_url,
                cluster_info=cluster_info,
                sample_fraction=sample_fraction,
                random_state=planner_random_state,
                pos_label=pos_label,
                split_col=split_col,
                sample_weight_col=sample_weight_col).generate(hyperparameters)

            trial = context.execute_trial(
                plan=trial_plan, metric=metric, flavor=MLFlowFlavor.SKLEARN)
            score = trial.metrics.get(metric.trial_metric_name, metric.worst_value)
            return {
                "loss": -score if metric.higher_is_better else score,
                "status": STATUS_OK,
                "trial_info": trial
            }

        return run_trial

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
            imputers: Dict[str, Imputer],
            num_exclude_cols: int,
            identity_col: Optional[Union[str, list]] = None,  # forecasting only
            horizon: Optional[int] = None,  # forecasting only
            frequency: Optional[str] = None,  # forecasting only
            output_database: Optional[str] = None,  # forecasting only
            pos_label: Optional[ClassificationTargetTypes] = None,  # binary classification only
    ) -> AutoMLSummary:

        no_semantic_type_detection_cols = list(imputers.keys())
        (data_source, cluster_info, preprocess_result, data_exploration_start_time, timeout,
         data_exp_nb_url, pos_label, split_col, sample_weight_col) = self._preprocess_data_impl(
             dataset, target_col, context, data_dir, metric, timeout, time_col, alert_manager,
             run_id, no_semantic_type_detection_cols, num_exclude_cols, pos_label)

        # Time check after running data exploration notebook
        training_start_time, timeout = self._time_check(data_exploration_start_time, timeout,
                                                        alert_manager)

        (trials, is_early_stopped) = self._train_impl(
            parallelism, context, metric, data_source, preprocess_result, target_col, time_col,
            cluster_info, training_start_time, timeout, max_trials, exclude_frameworks, imputers,
            alert_manager, pos_label, split_col, sample_weight_col)

        context.display_html(
            self._get_summary_html(
                trial=trials[0],
                data_exp_url=data_exp_nb_url,
                experiment_url=context.get_experiment_url(sort_metric=metric),
                metric=metric,
                sample_fraction=preprocess_result.size_estimator_result.sample_fraction))
        return AutoMLSummary(
            preprocess_result=preprocess_result,
            experiment=context.get_experiment(),
            trials=trials,
            semantic_type_conversions=preprocess_result.strong_semantic_detections,
            is_early_stopped=is_early_stopped)

    def _preprocess_data_impl(self,
                              dataset: DataFrame,
                              target_col: str,
                              context: Context,
                              data_dir: Optional[str],
                              metric: Metric,
                              timeout: Optional[int],
                              time_col: Optional[str],
                              alert_manager: AlertManager,
                              run_id: str,
                              no_semantic_type_detection_cols: List[str],
                              num_exclude_cols: int,
                              pos_label: Optional[ClassificationTargetTypes] = None
                              ) -> SupervisedLearnerDataPreprocessResults:
        """Preprocess the data before training.

        Data preprocessing consists the following main steps:
        1. Get pre_sampling_stats and drop invalid rows.
        2. Semantic detection.
        3. Sample the dataset.
        4. Save data snapshot.
        5. Run data exploration notebook.
        6. Get post_sampling_stats. Combine with results of previous stages to produce final preprocess_result.
        """

        # Start recording wall clock time
        start_time = datetime.now()

        cluster_info = self._validate_cluster()

        # Step 1. get pre_sampling_stats and drop invalid rows. These steps take full pass over the dataset.
        pre_sampling_stats = StatsCalculator.get_pre_sampling_stats(
            dataset=dataset,
            target_col=target_col,
            problem_type=self.problem_type,
        )

        log_supervised_input_data_stats(
            spark=self.spark,
            stats=pre_sampling_stats,
            run_id=run_id,
        )

        _logger.debug(f"DataStats calculated before sampling: {pre_sampling_stats}")

        pos_label = self._process_pos_label(pre_sampling_stats, target_col, alert_manager,
                                            pos_label)
        should_balance, target_label_ratios = self._should_balance(pre_sampling_stats,
                                                                   alert_manager, metric)

        dataset = self._drop_invalid_rows(
            dataset=dataset,
            target_col=target_col,
            dataset_stats=pre_sampling_stats,
            alert_manager=alert_manager,
        )

        # Step 2. Semantic detection.
        preprocessor_start_time, timeout = self._time_check(start_time, timeout, alert_manager)

        data_preprocessor = SupervisedLearnerDataPreprocessor(
            pre_sampling_stats=pre_sampling_stats,
            dataset_schema=dataset.schema,
            target_col=target_col,
            supported_target_types=self._get_supported_target_types(),
            time_col=time_col,
            supported_time_types=self._get_supported_time_types(),
            confs=self._confs,
            alert_manager=alert_manager,
        )
        feature_schema = data_preprocessor.feature_schema

        # Semantic type detection happens before sampling because sampling needs information from semantic
        # detection to estimate the memory. The detection is run on a sample of dataset if the original
        # dataset is too large.
        data_preprocessor.detect_semantic_types(
            dataset=dataset, no_detection_cols=no_semantic_type_detection_cols)
        strong_semantic_detections = data_preprocessor.strong_semantic_detections
        unsupported_cols = data_preprocessor.feature_schema.get(data_preprocessor.UNSUPPORTED_TYPE,
                                                                [])

        _logger.debug(f"Weak semantic detections: {data_preprocessor.weak_semantic_detections}")
        _logger.debug(f"Strong semantic detections: {data_preprocessor.strong_semantic_detections}")

        if unsupported_cols:
            alert_manager.record(UnsupportedColumnAlert(unsupported_cols))
            _logger.warning(
                f"The following columns are of unsupported data types and will be dropped: {unsupported_cols}"
            )

        # Step 3. Sample the dataset if necessary
        size_estimator_result = SizeEstimator(spark=self.spark).get_sampling_fraction(
            stats=pre_sampling_stats,
            strong_semantic_detections=strong_semantic_detections,
        )
        sample_fraction = size_estimator_result.sample_fraction

        split_col = None
        sample_weight_col = None
        if InternalConfs.ENABLE_TRAIN_TEST_SPLIT_DRIVER:
            split_col = f"{DataSplitter.SPLIT_COL_PREFIX}_{str(uuid.uuid4())[:4]}"

            # When time column is present, rows with null time column values need to be removed before split
            if time_col:
                dataset = self._process_time_col(dataset, time_col, pre_sampling_stats.num_rows,
                                                 data_preprocessor.time_col_type,
                                                 strong_semantic_detections, alert_manager)

            if should_balance:
                sample_weight_col = f"{self.SAMPLE_WEIGHT_COL_PREFIX}_{str(uuid.uuid4())[:4]}"
                data_preprocessor.selected_cols.append(sample_weight_col)

            dataset = self._add_split_column_and_maybe_sample(
                dataset=dataset,
                target_col=target_col,
                split_col=split_col,
                time_col=time_col,
                fraction=sample_fraction,
                pre_sampling_stats=pre_sampling_stats,
                alert_manager=alert_manager,
                sample_weight_col=sample_weight_col)

            data_preprocessor.selected_cols.append(split_col)
        else:
            if sample_fraction:
                dataset = self._sample(
                    dataset,
                    sample_fraction,
                    target_col=target_col,
                    dataset_stats=pre_sampling_stats,
                    alert_manager=alert_manager)

            # Time column processing should use semantic detection to ensure the time column contains
            # only datetime data, so the processing must happen after semantic type detection
            if time_col:
                dataset = self._process_time_col(dataset, time_col, pre_sampling_stats.num_rows,
                                                 data_preprocessor.time_col_type,
                                                 strong_semantic_detections, alert_manager)

        if sample_fraction:
            _logger.info(f"Dataset sampled to {sample_fraction * 100} % to allow "
                         "parallel trials to fit the dataset into memory.")

            context.set_sample_fraction(sample_fraction)
            alert_manager.record(DatasetTooLargeAlert())

        # Step 4. Save data snapshot.
        save_data_start_time, timeout = self._time_check(preprocessor_start_time, timeout,
                                                         alert_manager)

        data_source, dataset = context.save_training_data(
            dataset=dataset,
            data_dir=data_dir,
            data_format=DatasetFormat.PANDAS,
            selected_cols=data_preprocessor.selected_cols,
        )
        self._data_source = data_source

        _logger.debug(f"Saved the dataset to {self._data_source}")

        # Step 5. Run data exploration notebook.
        data_exploration_start_time, timeout = self._time_check(save_data_start_time, timeout,
                                                                alert_manager)
        data_exp_nb_details, data_exp_result = self._run_data_exploration(
            context=context,
            data_source=data_source,
            num_rows=dataset.count(),
            feature_schema=feature_schema,
            strong_semantic_detections=strong_semantic_detections,
            target_col=target_col,
            time_col=time_col,
            timeout=timeout,
            experiment_url=context.get_experiment_url(sort_metric=metric),
            cluster_info=cluster_info,
            sample_fraction=sample_fraction,
            internal_cols=[split_col, sample_weight_col],
            alert_manager=alert_manager,
        )

        _logger.info("Data exploration complete. Notebook can be found at: "
                     f"{data_exp_nb_details.url}")
        context.set_experiment_exploration_notebook(data_exp_nb_details.id)

        if data_exp_result:
            data_preprocessor.log_feature_alerts(data_exp_result)
        else:
            alert_manager.record(DataExplorationFailAlert())

        # Step 6. Get post_sampling_stats
        # Combine with results of previous stages to produce final preprocess_result.
        post_sampling_stats = StatsCalculator.get_post_sampling_stats(
            dataset,
            target_col,
            self.problem_type,
            strong_semantic_detections,
            pre_sampling_stats,
            exclude_cols=[split_col, sample_weight_col],
        )

        _logger.debug(f"DataStats calculated post sampling: {post_sampling_stats}")

        preprocess_result = data_preprocessor.process_post_sampling_stats(
            post_sampling_stats, target_col, self.problem_type, size_estimator_result)

        _logger.debug(f"Data Processing Results: {preprocess_result}")

        log_supervised_data_stats(
            spark=self.spark,
            notebook_id=data_exp_nb_details.id,
            num_target_nulls=pre_sampling_stats.num_target_nulls,
            preprocess_result=preprocess_result,
            post_sampling_stats=post_sampling_stats,
            num_exclude_cols=num_exclude_cols,
            num_array_columns=len(preprocess_result.array_columns),
            dataset_size_bytes=data_exp_result["table"]["memory_size"] if data_exp_result else 0,
            is_imbalanced=should_balance,
            target_label_ratios=target_label_ratios,
            run_id=run_id,
        )

        print(
            self._get_post_cmd_instructions(
                experiment_url=context.get_experiment_url(sort_metric=metric, absolute=True),
                metric=metric))

        return data_source, cluster_info, preprocess_result, data_exploration_start_time, timeout,\
               data_exp_nb_details.url, pos_label, split_col, sample_weight_col

    def _train_impl(
            self, parallelism: int, context: Context, metric: Metric, data_source: DataSource,
            preprocess_result: SupervisedLearnerDataPreprocessResults, target_col: str,
            time_col: Optional[str], cluster_info: Dict[str, str], training_start_time: datetime,
            timeout: Optional[int], max_trials: int, exclude_frameworks: Set[Framework],
            imputers: Dict[str, Imputer], alert_manager: AlertManager,
            pos_label: Optional[ClassificationTargetTypes], split_col: Optional[str],
            sample_weight_col: Optional[str]) -> Tuple[List[TrialInfo], bool]:
        experiment_url = context.get_experiment_url(sort_metric=metric)

        # Construct the hyper-parameter search space for the trials
        if exclude_frameworks:
            planners = [
                planner for planner in self._get_planners()
                if planner.framework() not in exclude_frameworks
            ]
        else:
            planners = self._get_planners()
        search_space = self._get_search_space(planners)

        # Set the random state seed used by planners and their generated code sections
        planner_random_state = np.random.randint(1e9)

        # Set the random state seed used by hyperopt to search the hyperparameter space
        hyperopt_random_state = np.random.randint(1e9)

        spark_trials = SparkTrials(parallelism=parallelism)

        run_trial_fn = self.get_run_trial_fn(
            context, metric, data_source, preprocess_result, target_col, time_col, imputers,
            experiment_url, cluster_info, preprocess_result.size_estimator_result.sample_fraction,
            planner_random_state, pos_label, split_col, sample_weight_col)

        try:
            fmin(
                fn=run_trial_fn,
                space=search_space,
                algo=hyperopt.tpe.suggest,
                max_evals=max_trials,
                trials=spark_trials,
                timeout=timeout,
                rstate=np.random.default_rng(hyperopt_random_state),
                early_stop_fn=get_early_stop_fn(
                    no_early_stop_threshold=NO_EARLY_STOP_THRESHOLD,
                    no_progress_stop_threshold=NO_PROGRESS_STOP_THRESHOLD),
                show_progressbar=False,  # Disable progress bar because we set a large max_evals
            )
        except Exception as e:
            # https://github.com/hyperopt/hyperopt/blob/0.2.5/hyperopt/fmin.py#L558
            if str(e) == "There are no evaluation tasks, cannot return argmin of task losses.":
                # Time check to see if we have hit a timeout or not
                post_training_time, timeout = self._time_check(training_start_time, timeout,
                                                               alert_manager)

                # If no timeout, then this is a failure with trials
                raise ExecutionResultError(
                    "All trials either failed or did not return results to hyperopt.")
            else:
                raise e
        is_early_stopped = (
            spark_trials.fmin_cancelled_reason == FMIN_CANCELLED_REASON_EARLY_STOPPING)
        if is_early_stopped:
            alert_manager.record(EarlyStopAlert())
        trials = [trial_result["trial_info"] for trial_result in spark_trials.results]

        self._clean_failed_mlflow_runs(context.experiment_id, data_source, trials)

        # Sort trials by val metric, so best trial is first
        trials = self._sort_trials(trials, metric)
        best_trial = trials[0]
        context.set_experiment_best_trial_notebook(best_trial.notebook_id)

        return trials, is_early_stopped

    @staticmethod
    def _process_time_col(dataset: DataFrame, time_col: str, num_rows: int,
                          time_type: Type[DataType],
                          strong_semantic_detections: Dict[SemanticType, List[str]],
                          alert_manager: AlertManager) -> DataFrame:
        if time_col in strong_semantic_detections[SemanticType.DATETIME]:
            psdf = dataset.to_pandas_on_spark()
            psdf[time_col] = ps.to_datetime(
                psdf[time_col], infer_datetime_format=True, errors="coerce")
            dataset = psdf.to_spark()
        elif time_col in strong_semantic_detections[SemanticType.NUMERIC]:
            psdf = dataset.to_pandas_on_spark()
            psdf[time_col] = ps.to_numeric(psdf[time_col])
            dataset = psdf.to_spark()
        elif time_type == StringType:
            alert_manager.record(UnsupportedTimeTypeAlert(time_col, str(time_type)))
            raise UnsupportedColumnError(
                f"Column \"{time_col}\" cannot be cast to timestamp/int, it cannot be used for train-val-test split."
            )

        dataset = dataset.filter(col(time_col).isNotNull())
        num_rows_left = dataset.count()
        if num_rows_left == 0:
            raise UnsupportedDataError(
                f"The input dataset is empty after dropping nulls in selected time_col {time_col}. "
                "Please pass in a valid dataset.")
        num_time_nulls = num_rows - num_rows_left
        if num_time_nulls != 0:
            _logger.warning(
                f"The selected time_col {time_col} includes {num_time_nulls} null values. "
                "AutoML will drop all rows with these null values.")
            alert_manager.record(NullsInTimeColumnAlert(time_col, str(time_type)))
        return dataset

    def _sample(self,
                dataset: DataFrame,
                fraction: float,
                target_col: str,
                dataset_stats: PreSamplingStats,
                alert_manager: AlertManager,
                min_rows_to_ensure=5) -> DataFrame:
        seed = np.random.randint(1e9)
        return dataset.sample(fraction=fraction, seed=seed)

    def _sample_and_balance(self, dataset: DataFrame, fraction: float, target_col: str,
                            dataset_stats: PreSamplingStats, sample_weight_col: str) -> DataFrame:
        """
        Sample classification data with balanced ratio among classes.
        """
        return dataset

    def _run_data_exploration(
            self,
            context: Context,
            data_source: DataSource,
            num_rows: int,
            feature_schema: Dict[Union[str, Type[DataType]], List[str]],
            strong_semantic_detections: Dict[SemanticType, List[str]],
            target_col: str,
            time_col: Optional[str],
            timeout: Optional[int],
            experiment_url: str,
            cluster_info: Dict[str, str],
            sample_fraction: Optional[float],
            internal_cols: List[Optional[str]],
            alert_manager: AlertManager,
    ) -> Tuple[NotebookDetails, Dict[str, Any]]:

        num_cols = sum([len(value) for value in feature_schema.values()])
        # generate and run the data exploration notebook
        data_exploration_plan = Plan(
            name="DataExploration",
            sections=[
                InputDataExplorationSupervisedLearner(
                    data_source=data_source,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    date_cols=feature_schema[DateType],
                    numerical_cols=feature_schema[SupervisedLearnerDataPreprocessor.NUMERICAL_TYPE],
                    target_col=target_col,
                    strong_semantic_detections=strong_semantic_detections,
                    experiment_url=experiment_url,
                    driver_notebook_url=context.driver_notebook_url,
                    cluster_info=cluster_info,
                    problem_type=self.problem_type,
                    sample_fraction=sample_fraction,
                    alert_manager=alert_manager,
                    time_col=time_col,
                    internal_cols=list(filter(lambda c: c is not None, internal_cols)),
                )
            ])
        return context.execute_data_exploration(
            plan=data_exploration_plan, data_source=data_source, timeout=timeout)

    @staticmethod
    def _time_check(start_time: datetime, timeout: Optional[int],
                    alert_manager: AlertManager) -> Tuple[datetime, Optional[int]]:
        """
        Checks if the user has passed a non-null timeout and recalculates the new timeout
        based on the elapsed time. If the new timeout is <= 0, throw an ExecutionTimeoutError
        to indicate that we have reached a timeout

        :param start_time: Time when AutoML run started
        :param timeout: Optional timeout passed by the user
        :returns (Current time, Newly calculated timeout or None)
        """
        now = datetime.now()
        if not timeout:
            return now, timeout

        elapsed_time = int((now - start_time).total_seconds())
        new_timeout = timeout - elapsed_time
        if new_timeout <= 0:
            alert_manager.record(ExecutionTimeoutAlert())
            raise ExecutionTimeoutError(
                "Execution timed out before any trials could be successfully run. "
                "Please increase the timeout for AutoML to run some trials.")
        return now, new_timeout

    def _process_pos_label(self,
                           pre_sampling_stats: PreSamplingStats,
                           target_col: str,
                           alert_manager: AlertManager,
                           pos_label: Optional[ClassificationTargetTypes] = None
                           ) -> Optional[ClassificationTargetTypes]:
        return pos_label

    def _should_balance(self, pre_sampling_stats: PreSamplingStats, alert_manager: AlertManager,
                        metric: Metric) -> Tuple[bool, Optional[List[float]]]:
        """
        Determine whether the target labels are imbalanced for classification problem, and return target label ratios.
        """
        return False, None

    def _add_split_column_and_maybe_sample(self,
                                           dataset: DataFrame,
                                           target_col: str,
                                           split_col: str,
                                           time_col: Optional[str],
                                           fraction: Optional[float],
                                           pre_sampling_stats: PreSamplingStats,
                                           alert_manager: AlertManager,
                                           sample_weight_col: Optional[str] = None) -> DataFrame:

        # Split dataset into train/val/test data in fractions [0.6, 0.2, 0.2]
        rng = np.random.default_rng(2022)
        seed = int(rng.integers(1e9))

        train_df, val_df, test_df = self.splitter.split(
            df=dataset,
            target_col=target_col,
            ratios=[0.6, 0.2, 0.2],
            time_col=time_col,
            class_counts=pre_sampling_stats.class_counts,
            seed=seed)

        # Sample training data
        if sample_weight_col and fraction:
            train_df = self._sample_and_balance(train_df, fraction, target_col, pre_sampling_stats,
                                                sample_weight_col)
        elif sample_weight_col and not fraction:
            # When an imbalanced dataset is not as large as to sampled to fit in memory,
            # still down-sample with a default fraction
            train_df = self._sample_and_balance(train_df, self.BALANCED_SAMPLE_FRACTION, target_col,
                                                pre_sampling_stats, sample_weight_col)
        elif fraction:
            train_df = self._sample(
                train_df,
                fraction,
                target_col,
                pre_sampling_stats,
                alert_manager,
                min_rows_to_ensure=StatsCalculator.MIN_ROWS_PER_LABEL_AFTER_SPLIT)

        # Sample val / test data
        if fraction:
            val_df = self._sample(
                val_df,
                fraction,
                target_col,
                pre_sampling_stats,
                alert_manager,
                min_rows_to_ensure=StatsCalculator.MIN_ROWS_PER_LABEL_AFTER_SPLIT)
            test_df = self._sample(
                test_df,
                fraction,
                target_col,
                pre_sampling_stats,
                alert_manager,
                min_rows_to_ensure=StatsCalculator.MIN_ROWS_PER_LABEL_AFTER_SPLIT)

        # Keep the schema of each split the same
        if sample_weight_col:
            val_df = val_df.withColumn(sample_weight_col, lit(1))
            test_df = test_df.withColumn(sample_weight_col, lit(1))

        # Add split column
        train_df = train_df.withColumn(split_col, lit("train"))
        val_df = val_df.withColumn(split_col, lit("val"))
        test_df = test_df.withColumn(split_col, lit("test"))

        return train_df.union(val_df.union(test_df))
