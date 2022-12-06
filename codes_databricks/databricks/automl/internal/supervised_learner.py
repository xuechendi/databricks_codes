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

from databricks.automl.internal.alerts import DatasetTooLargeAlert, EarlyStopAlert, UnsupportedTimeTypeAlert, \
    DataExplorationFailAlert, NullsInTimeColumnAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.base_learner import BaseLearner
from databricks.automl.internal.classification.stats import ClassificationStatsCalculator, \
    ClassificationIntermediateStats
from databricks.automl.internal.common.const import ContextType, \
    DatasetFormat, Framework, SparkDataType
from databricks.automl.internal.confs import InternalConfs
from databricks.automl.internal.context import Context, DataSource, NotebookDetails
from databricks.automl.internal.data_augmentation.feature_store import FeatureStoreJoiner
from databricks.automl.internal.data_splitter import DataSplitter, RandomDataSplitter
from databricks.automl.internal.errors import ExecutionResultError, UnsupportedColumnError
from databricks.automl.internal.imputers import Imputer
from databricks.automl.internal.instrumentation import instrumented, log_classification_input_data_stats, \
    log_supervised_data_stats, log_supervised_input_data_stats
from databricks.automl.internal.plan import Plan
from databricks.automl.internal.preprocess import SupervisedLearnerDataPreprocessor, \
    SupervisedLearnerDataPreprocessConfs, \
    SupervisedLearnerDataPreprocessResults
from databricks.automl.internal.sections.exploration.data import InputDataExplorationSupervisedLearner
from databricks.automl.internal.semantic_detector import SemanticDetector
from databricks.automl.internal.size_estimator import SizeEstimator
from databricks.automl.internal.stats import StatsCalculator, IntermediateStats, InputStats
from databricks.automl.internal.utils import time_check
from databricks.automl.shared.cell_output import CellOutput
from databricks.automl.shared.const import ClassificationTargetTypes, Metric, MLFlowFlavor, SemanticType, ProblemType
from databricks.automl.shared.errors import UnsupportedDataError
from databricks.automl.shared.result import AutoMLSummary, TrialInfo

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
        return SparkDataType.TIME_TYPES + \
               SparkDataType.INTEGER_TYPES + \
               SparkDataType.STRING_TYPE

    @property
    def splitter(self) -> DataSplitter:
        return RandomDataSplitter()

    @property
    def supported_frameworks(self) -> List[Framework]:
        return [Framework.SKLEARN, Framework.XGBOOST, Framework.LIGHTGBM]

    @instrumented
    def fit(
            self,
            dataset: Union[DataFrame, pd.DataFrame, ps.DataFrame, str],
            *,
            target_col: str,
            data_dir: Optional[str] = None,
            exclude_cols: List[str] = [],
            exclude_columns: List[str] = [],
            exclude_frameworks: List[str] = None,
            feature_store_lookups: Optional[List[Dict]] = None,
            imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
            # max_trials is deprecated and only checked for deprecation warning
            max_trials: Optional[int] = None,
            timeout_minutes: Optional[int] = None,
            time_col: Optional[str] = None,
            pos_label: Optional[ClassificationTargetTypes] = None,
            experiment: Optional[mlflow.entities.Experiment] = None,
            home_dir: Optional[str] = None,
            metric: Optional[Union[Metric, str]] = None,
            parallelism: int = BaseLearner.MAX_TRIAL_PARALLELISM,
            run_id: Optional[str] = None) -> AutoMLSummary:
        """
        For user-facing parameters, which are not documented here, see docstrings in __init__.py
        :param experiment: MLflow experiment to log this AutoML run to. This experiment should be new and unused.
                           If no experiment is given, a brand new one will be created.
        :param home_dir: This is the same as experiment_dir but with a different name to keep backwards compatibility
                         with the usage of this internal API by our AutoML UI:
                         https://github.com/databricks/universe/blob/master/webapp/web/js/mlflow/autoML/AutoMLDriverNotebook.ipynb
        :param metric: This is the same as primary_metric in the user-facing API.
        :param parallelism: The maximum parallelism to use when running trials.
                            The actual parallelism is subject to available Spark task slots at
                            runtime.
                            If set to None (default) or a non-positive value, this will be set to
                            Spark's default parallelism or `1`.
                            We cap the value at `MAX_CONCURRENT_JOBS_ALLOWED=128`.
        :param run_id: UUID set by @instrumented decorator. Do not set this manually.
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
            pos_label=pos_label,
            feature_store_lookups=feature_store_lookups)

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
                metric=metric.trial_metric_name,
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
            feature_store_lookups: Optional[List[Dict]] = None) -> AutoMLSummary:

        no_semantic_type_detection_cols = list(imputers.keys())
        (data_source, cluster_info, preprocess_result, data_exploration_start_time, timeout,
         data_exp_nb_url, pos_label, split_col, sample_weight_col) = self._preprocess_data_impl(
             dataset, target_col, context, data_dir, metric, timeout, time_col, alert_manager,
             run_id, no_semantic_type_detection_cols, num_exclude_cols, pos_label,
             feature_store_lookups)

        # Time check after running data exploration notebook
        training_start_time, timeout = time_check(data_exploration_start_time, timeout,
                                                  alert_manager)

        (trials, is_early_stopped) = self._train_impl(
            parallelism, context, metric, data_source, preprocess_result, target_col, time_col,
            cluster_info, training_start_time, timeout, max_trials, exclude_frameworks, imputers,
            alert_manager, pos_label, split_col, sample_weight_col)

        context.display_html(
            CellOutput.get_summary_html(
                trial=trials[0],
                data_exp_url=data_exp_nb_url,
                experiment_url=context.get_experiment_url(),
                metric=metric,
                problem_type=self.problem_type,
                sample_fraction=preprocess_result.size_estimator_result.precise_sample_fraction))
        return AutoMLSummary(
            experiment_id=context.get_experiment().experiment_id,
            trials=trials,
            semantic_type_conversions=preprocess_result.strong_semantic_detections,
            is_early_stopped=is_early_stopped)

    def _preprocess_data_impl(
            self,
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
            pos_label: Optional[ClassificationTargetTypes] = None,
            feature_store_lookups: Optional[List[Dict]] = None,
    ):
        """Preprocess the data before training.

        Data preprocessing consists the following main steps:
        1. Get input_stats and log it. Drop invalid rows.
        2. Join features from Feature Store as specified. This step might take a rough sampling to avoid
           unnecessary long joins.
        3. Cast Decimal datatype to float64.
        4. Get intermediate_stats
        5. Semantic detection.
        6. Sample the dataset if necessary.
        7. Save data snapshot.
        8. Run data exploration notebook.
        9. Get post_sampling_stats. Combine with results of previous stages to produce final preprocess_result.
        """

        # Start recording wall clock time
        start_time = datetime.now()

        cluster_info = self._validate_cluster()

        # Step 1. Get and log input stats and drop invalid rows.
        # These steps take full pass over the dataset.
        # TODO (ML-24173) split fit for each problem type
        if self.problem_type == ProblemType.CLASSIFICATION:
            stats_calculator = ClassificationStatsCalculator(
                target_col=target_col,
                time_col=time_col,
                run_id=run_id,
                pos_label=pos_label,
                metric=metric)
        else:
            stats_calculator = StatsCalculator(
                target_col=target_col, time_col=time_col, run_id=run_id)

        input_stats = stats_calculator.get_input_stats(dataset=dataset)

        # TODO (ML-24173) split fit for each problem type
        if self.problem_type == ProblemType.CLASSIFICATION:
            log_classification_input_data_stats(
                spark=self.spark,
                stats=input_stats,
                run_id=run_id,
            )
        else:
            log_supervised_input_data_stats(
                spark=self.spark,
                stats=input_stats,
                run_id=run_id,
            )

        dataset = self._drop_invalid_rows(
            dataset=dataset,
            target_col=target_col,
            dataset_stats=input_stats,
            alert_manager=alert_manager,
        )
        self._validate_dataset_has_rows(target_col, input_stats, alert_manager)

        # Step 2. Join features from Feature Store if specified
        size_estimator = SizeEstimator(spark=self.spark)
        has_feature_store_joins = False
        feature_spec = None
        rough_sample_fraction = None
        if feature_store_lookups:
            joiner = FeatureStoreJoiner(feature_store_lookups)
            num_features = joiner.get_num_features()
            rough_sample_fraction = size_estimator.get_rough_sampling_fraction(
                input_stats, num_features)
            if rough_sample_fraction:
                dataset = self._sample(
                    dataset,
                    rough_sample_fraction,
                    target_col=target_col,
                    dataset_stats=input_stats,
                    alert_manager=alert_manager)
            dataset, feature_spec, column_renames, joined_cols = joiner.join_features(
                dataset, target_col)
            has_feature_store_joins = True
            no_semantic_type_detection_cols += list(joined_cols)

        # Step 3. Cast Decimal type to float64 type.
        dataset = dataset.select([
            col(f.name).cast("double").alias(f.name)
            if isinstance(f.dataType, SparkDataType.DECIMAL_TYPE) else col(f.name)
            for f in dataset.schema.fields
        ])

        # Step 4. Calculate IntermediateStats, which is used in semantic detection and precise sampling and
        # the data preprocessor.
        intermediate_stats = stats_calculator.get_intermediate_stats(
            dataset=dataset,
            input_num_rows=input_stats.num_rows - input_stats.num_invalid_rows,
            is_sampled=(feature_store_lookups is not None) and (rough_sample_fraction is not None))
        should_balance, target_label_ratios, pos_label = \
            stats_calculator.validate_intermediate_stats(intermediate_stats, alert_manager)

        # Step 5. Semantic detection.
        preprocessor_start_time, timeout = time_check(start_time, timeout, alert_manager)

        data_preprocessor = SupervisedLearnerDataPreprocessor(
            intermediate_stats=intermediate_stats,
            dataset_schema=dataset.schema,
            target_col=target_col,
            supported_target_types=self._get_supported_target_types(),
            time_col=time_col,
            supported_time_types=self._get_supported_time_types(),
            confs=self._confs,
            alert_manager=alert_manager,
        )

        # Semantic type detection happens before sampling because sampling needs information from semantic
        # detection to estimate the memory. The detection is run on a sample of dataset if the original
        # dataset is too large.
        semantic_detector = SemanticDetector(
            intermediate_stats=intermediate_stats,
            target_col=target_col,
            alert_manager=alert_manager)
        strong_semantic_detections, _ = semantic_detector.detect_semantic_types(
            dataset=dataset, no_detection_cols=no_semantic_type_detection_cols)

        # Step 6. Sample the dataset if necessary
        size_estimator_result = SizeEstimator(spark=self.spark).get_sampling_fraction(
            stats=intermediate_stats,
            strong_semantic_detections=strong_semantic_detections,
            rough_sample_fraction=rough_sample_fraction)
        precise_sample_fraction = size_estimator_result.precise_sample_fraction

        split_col = None
        sample_weight_col = None
        # TODO: remove unnecessary InternalConfs
        if InternalConfs.ENABLE_TRAIN_TEST_SPLIT_DRIVER:
            split_col = f"{DataSplitter.SPLIT_COL_PREFIX}_{str(uuid.uuid4())[:4]}"

            # When time column is present, rows with null time column values need to be removed before split
            time_col_to_split_on = None
            if time_col:
                dataset, time_col_to_split_on = self._process_time_col(
                    dataset, time_col, intermediate_stats.num_rows, data_preprocessor.time_col_type,
                    strong_semantic_detections, alert_manager)

            if should_balance:
                sample_weight_col = f"{self.SAMPLE_WEIGHT_COL_PREFIX}_{str(uuid.uuid4())[:4]}"
                intermediate_stats.supported_cols.append(sample_weight_col)

            dataset = self._add_split_column_and_maybe_sample(
                dataset=dataset,
                target_col=target_col,
                split_col=split_col,
                time_col=time_col_to_split_on,
                fraction=precise_sample_fraction,
                intermediate_stats=intermediate_stats,
                alert_manager=alert_manager,
                sample_weight_col=sample_weight_col)

            intermediate_stats.supported_cols.append(split_col)
        else:
            if precise_sample_fraction:
                dataset = self._sample(
                    dataset,
                    precise_sample_fraction,
                    target_col=target_col,
                    dataset_stats=intermediate_stats,
                    alert_manager=alert_manager)

            # Time column processing should use semantic detection to ensure the time column contains
            # only datetime data, so the processing must happen after semantic type detection
            if time_col:
                dataset, _ = self._process_time_col(dataset, time_col, intermediate_stats.num_rows,
                                                    data_preprocessor.time_col_type,
                                                    strong_semantic_detections, alert_manager)

        final_sample_fraction = size_estimator_result.final_sample_fraction
        if final_sample_fraction:
            _logger.info(f"Dataset sampled to {final_sample_fraction * 100} % to allow "
                         "parallel trials to fit the dataset into memory.")

            context.set_sample_fraction(final_sample_fraction)
            alert_manager.record(DatasetTooLargeAlert())

        # Step 7. Save data snapshot.
        save_data_start_time, timeout = time_check(preprocessor_start_time, timeout, alert_manager)

        data_source, dataset = context.save_training_data(
            dataset=dataset,
            data_dir=data_dir,
            data_format=DatasetFormat.PANDAS,
            selected_cols=intermediate_stats.supported_cols,
            feature_spec=feature_spec,
        )
        self._data_source = data_source

        _logger.debug(f"Saved the dataset to {self._data_source}")

        # Step 8. Run data exploration notebook.
        data_exploration_start_time, timeout = time_check(save_data_start_time, timeout,
                                                          alert_manager)
        data_exp_nb_details, data_exp_result = self._run_data_exploration(
            context=context,
            data_source=data_source,
            num_rows=dataset.count(),
            feature_schema=intermediate_stats.feature_schema,
            strong_semantic_detections=strong_semantic_detections,
            target_col=target_col,
            time_col=time_col,
            timeout=timeout,
            experiment_url=context.get_experiment_url(),
            cluster_info=cluster_info,
            sample_fraction=final_sample_fraction,
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

        # Step 9. Get post_sampling_stats
        # Combine with results of previous stages to produce final preprocess_result.
        post_sampling_stats = StatsCalculator.get_post_sampling_stats(
            dataset,
            target_col,
            self.problem_type,
            strong_semantic_detections,
            intermediate_stats,
            exclude_cols=[split_col, sample_weight_col],
        )

        _logger.debug(f"DataStats calculated post sampling: {post_sampling_stats}")

        preprocess_result = data_preprocessor.process_post_sampling_stats(
            post_sampling_stats, target_col, self.problem_type, size_estimator_result)
        preprocess_result.strong_semantic_detections = strong_semantic_detections
        preprocess_result.has_feature_store_joins = has_feature_store_joins

        _logger.debug(f"Data Processing Results: {preprocess_result}")

        log_supervised_data_stats(
            spark=self.spark,
            notebook_id=data_exp_nb_details.id,
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
                experiment_url=context.get_experiment_url(absolute=True), metric=metric))

        return data_source, cluster_info, preprocess_result, data_exploration_start_time, timeout, \
               data_exp_nb_details.url, pos_label, split_col, sample_weight_col

    def _train_impl(
            self, parallelism: int, context: Context, metric: Metric, data_source: DataSource,
            preprocess_result: SupervisedLearnerDataPreprocessResults, target_col: str,
            time_col: Optional[str], cluster_info: Dict[str, str], training_start_time: datetime,
            timeout: Optional[int], max_trials: int, exclude_frameworks: Set[Framework],
            imputers: Dict[str, Imputer], alert_manager: AlertManager,
            pos_label: Optional[ClassificationTargetTypes], split_col: Optional[str],
            sample_weight_col: Optional[str]) -> Tuple[List[TrialInfo], bool]:
        experiment_url = context.get_experiment_url()

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
            experiment_url, cluster_info,
            preprocess_result.size_estimator_result.precise_sample_fraction, planner_random_state,
            pos_label, split_col, sample_weight_col)

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
                post_training_time, timeout = time_check(training_start_time, timeout,
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
                          alert_manager: AlertManager) -> Tuple[DataFrame, str]:
        """
        Processes the user-specified `time_col` in `dataset` by filtering out invalid dates and
        converting values to a datetime-compatible type if necessary. If values need conversion, the
        returned DataFrame will have a new column with the converted values.

        :param dataset:   The DataFrame being trained on.
        :param time_col:  The name of the user-specified time column to use for train/test/val split.
        :param num_rows:  The number of rows in `dataset`.
        :param time_type: The type of `time_col`.
        :param strong_semantic_detections: Dictionary of semantic type detections by type, used to
                                           convert `time_col` to a datetime-compatible type.
        :param alert_manager: AlertManager used to pass warnings to the user.
        :return: The new DataFrame object and the name the time column to use for splitting.
        """

        def _get_converted_time_col_name(time_col: str, col_names: List[str]) -> str:
            converted_time_col_name = time_col + "_converted"
            while converted_time_col_name in col_names:
                # If there's a column name conflict, append some random numbers
                converted_time_col_name += str(np.random.randint(100))
            return converted_time_col_name

        time_col_to_split_on = time_col
        if time_col in strong_semantic_detections[SemanticType.DATETIME]:
            psdf = dataset.to_pandas_on_spark()
            time_col_to_split_on = _get_converted_time_col_name(time_col, dataset.columns)
            psdf[time_col_to_split_on] = ps.to_datetime(
                psdf[time_col], infer_datetime_format=True, errors="coerce")
            dataset = psdf.to_spark()
        elif time_col in strong_semantic_detections[SemanticType.NUMERIC]:
            psdf = dataset.to_pandas_on_spark()
            time_col_to_split_on = _get_converted_time_col_name(time_col, dataset.columns)
            psdf[time_col_to_split_on] = ps.to_numeric(psdf[time_col])
            dataset = psdf.to_spark()
        elif time_type == StringType:
            alert_manager.record(UnsupportedTimeTypeAlert(time_col, str(time_type)))
            raise UnsupportedColumnError(
                f"Column \"{time_col}\" cannot be cast to timestamp/int, it cannot be used for train-val-test split."
            )

        dataset = dataset.filter(col(time_col_to_split_on).isNotNull())
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
        return dataset, time_col_to_split_on

    def _sample(self,
                dataset: DataFrame,
                fraction: float,
                target_col: str,
                dataset_stats: Union[IntermediateStats, InputStats],
                alert_manager: AlertManager,
                min_rows_to_ensure=5) -> DataFrame:
        seed = np.random.randint(1e9)
        return dataset.sample(fraction=fraction, seed=seed)

    def _sample_and_balance(self, dataset: DataFrame, fraction: float, target_col: str,
                            dataset_stats: IntermediateStats, sample_weight_col: str) -> DataFrame:
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
                    numerical_cols=feature_schema[SparkDataType.NUMERICAL_TYPE],
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

    def _should_balance(self, intermediate_stats: IntermediateStats, alert_manager: AlertManager,
                        metric: Metric) -> Tuple[bool, Optional[List[float]]]:
        """
        Determine whether the target labels are imbalanced for classification problem, and return target label ratios.
        """
        return False, None

    def _add_split_column_and_maybe_sample(
            self,
            dataset: DataFrame,
            target_col: str,
            split_col: str,
            time_col: Optional[str],
            fraction: Optional[float],
            intermediate_stats: Union[IntermediateStats, ClassificationIntermediateStats],
            alert_manager: AlertManager,
            sample_weight_col: Optional[str] = None) -> DataFrame:

        # Split dataset into train/val/test data in fractions [0.6, 0.2, 0.2]
        rng = np.random.default_rng(2022)
        seed = int(rng.integers(1e9))

        if hasattr(intermediate_stats, 'class_counts'):
            class_counts = intermediate_stats.class_counts
        else:
            class_counts = None

        train_df, val_df, test_df = self.splitter.split(
            df=dataset,
            target_col=target_col,
            ratios=[0.6, 0.2, 0.2],
            time_col=time_col,
            class_counts=class_counts,
            seed=seed)

        # Sample training data
        if sample_weight_col and fraction:
            train_df = self._sample_and_balance(train_df, fraction, target_col, intermediate_stats,
                                                sample_weight_col)
        elif sample_weight_col and not fraction:
            # When an imbalanced dataset is not as large as to sampled to fit in memory,
            # still down-sample with a default fraction
            train_df = self._sample_and_balance(train_df, self.BALANCED_SAMPLE_FRACTION, target_col,
                                                intermediate_stats, sample_weight_col)
        elif fraction:
            train_df = self._sample(
                train_df,
                fraction,
                target_col,
                intermediate_stats,
                alert_manager,
                min_rows_to_ensure=ClassificationStatsCalculator.MIN_ROWS_PER_LABEL_AFTER_SPLIT)

        # Sample val / test data
        if fraction:
            val_df = self._sample(
                val_df,
                fraction,
                target_col,
                intermediate_stats,
                alert_manager,
                min_rows_to_ensure=ClassificationStatsCalculator.MIN_ROWS_PER_LABEL_AFTER_SPLIT)
            test_df = self._sample(
                test_df,
                fraction,
                target_col,
                intermediate_stats,
                alert_manager,
                min_rows_to_ensure=ClassificationStatsCalculator.MIN_ROWS_PER_LABEL_AFTER_SPLIT)

        # Keep the schema of each split the same
        if sample_weight_col:
            val_df = val_df.withColumn(sample_weight_col, lit(1))
            test_df = test_df.withColumn(sample_weight_col, lit(1))

        # Add split column
        train_df = train_df.withColumn(split_col, lit("train"))
        val_df = val_df.withColumn(split_col, lit("val"))
        test_df = test_df.withColumn(split_col, lit("test"))

        return train_df.union(val_df.union(test_df))
