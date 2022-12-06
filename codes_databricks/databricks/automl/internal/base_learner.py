import logging
import textwrap
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable, Union

import mlflow
import pandas as pd
import pyspark.pandas as ps
from hyperopt import hp
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DataType, StructType

from databricks.automl.shared import utils as shared_utils
from databricks.automl.internal.alerts import DuplicateColumnNamesAlert, NullsInTargetColumnAlert, DatasetEmptyAlert, \
    AllRowsInvalidAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.shared.const import ClassificationTargetTypes
from databricks.automl.internal.common.const import ContextType, Framework, RunState

from databricks.automl.internal.context import ContextFactory, Context, DataSource
from databricks.automl.shared.errors import AutomlError, InvalidArgumentError, UnsupportedDataError
from databricks.automl.internal.errors import UnsupportedRuntimeError, UnsupportedClusterError
from databricks.automl.internal.imputers import Imputer
from databricks.automl.internal.planner import TrialPlanner
from databricks.automl.internal.stats import InputStats
from databricks.automl.shared.const import Metric, ProblemType
from databricks.automl.shared.result import AutoMLSummary, TrialInfo

_logger = logging.getLogger(__name__)

# ML-13588: Turn off the hyperopt warning. Remove after fixing hyperopt MLflow autologging issue.
_logger_hyperopt = logging.getLogger("hyperopt-spark")
_logger_hyperopt.setLevel(logging.ERROR)


def chain_invalid_argument_error(arg_name):
    """
    Use this decorator for functions that parse/validate user-provided arguments.
    Exceptions that are not AutomlError will be re-raised as InvalidArgumentError.
    :param arg_name: user-facing argument name defined in __init__.py API's
    """

    def decorator(func):
        def try_except_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AutomlError:
                raise  # re-raise the existing AutomlError
            except Exception as e:
                msg = (
                    f"Invalid argument `{arg_name}`. Run `help(databricks.automl.classify/regress/forecast)` "
                    "to read documentation for this argument.")
                raise InvalidArgumentError(msg) from e

        return try_except_wrapper

    return decorator


@dataclass
class ClusterEnvironment:
    """
    cluster envs from users
    """
    hyperopt_conf: str
    mlflow_run_ids: List[str]


class BaseLearner(ABC):
    DEFAULT_MAX_TRIALS = 10000
    DEFAULT_TIMEOUT_MINUTES = 120
    MAX_TRIAL_PARALLELISM = 100
    MINIMUM_TIMEOUT_MINUTES = 5  # minutes

    def __init__(self, context_type: ContextType):
        self._context_type = context_type
        self._mlflow_client = MlflowClient()
        self.spark = SparkSession.builder.getOrCreate()

    @property
    @abstractmethod
    def problem_type(self) -> ProblemType:
        pass

    @property
    @abstractmethod
    def default_metric(self) -> Metric:
        pass

    @classmethod
    @abstractmethod
    def supported_metrics(cls) -> Iterable[Metric]:
        pass

    @property
    @abstractmethod
    def supported_frameworks(self) -> List[Framework]:
        pass

    @staticmethod
    @chain_invalid_argument_error("exclude_columns")
    def _filter_exclude_cols(dataset: DataFrame, exclude_cols: List[str], target_col: str,
                             time_col: Optional[str]) -> Set[str]:
        """
        Validates the list of excluded columns provided by the user, and returns a sanitized list.
        Do not need to check if all columns get excluded. StatsCalculator._get_schema_info
        will check this and throw an exception.
        :param dataset: Spark DataFrame
        :param exclude_cols: user-specified columns that should be dropped from the dataset
        :param target_col: target column, which cannot be excluded
        :param time_col: if this provided by the user, it cannot be excluded
        :return: a deduplicated list of columns that can be dropped from the dataset
        """
        exclude_cols = set(exclude_cols)
        columns = set(dataset.columns)

        valid_exclude_cols = exclude_cols & columns
        non_existent_exclude_cols = exclude_cols - columns

        if len(non_existent_exclude_cols) > 0:
            _logger.warning(
                f"Some exclude_cols are not columns in the dataset: {non_existent_exclude_cols}")

        if target_col in valid_exclude_cols:
            _logger.warning(
                f"Target column {target_col} cannot be excluded. Skipping this exclusion.")
            valid_exclude_cols.remove(target_col)

        if time_col is not None and time_col in valid_exclude_cols:
            _logger.warning(f"Time column {time_col} cannot be excluded. Skipping this exclusion.")
            valid_exclude_cols.remove(time_col)

        return valid_exclude_cols

    @staticmethod
    def _validate_column_uniqueness(dataset: DataFrame, alert_manager: AlertManager):
        columns = [c for c, t in dataset.dtypes]
        duplicate_columns = [column for column, count in Counter(columns).items() if count > 1]
        if len(duplicate_columns) > 0:
            alert_manager.record(DuplicateColumnNamesAlert(duplicate_columns))
            raise UnsupportedDataError(
                f"Duplicate column names are not allowed: {duplicate_columns}")

    @chain_invalid_argument_error("exclude_frameworks")
    def _validate_and_filter_exclude_frameworks(self,
                                                exclude_frameworks: List[str]) -> Set[Framework]:
        exclude_frameworks = set(exclude_frameworks)
        # We need to cast the Enum value to string to compare with the user input.
        supported_frameworks = {x.value for x in self.supported_frameworks}

        valid_exclude_frameworks = exclude_frameworks & supported_frameworks
        non_existent_exclude_frameworks = exclude_frameworks - supported_frameworks
        valid_frameworks = supported_frameworks - exclude_frameworks

        if len(valid_frameworks) == 0:
            raise InvalidArgumentError(
                f"All supported frameworks are excluded: {supported_frameworks}")
        if len(non_existent_exclude_frameworks) > 0:
            _logger.warning(
                f"Some exclude_frameworks are not supported: {non_existent_exclude_frameworks}")

        return {Framework(x) for x in valid_exclude_frameworks}

    @staticmethod
    @chain_invalid_argument_error("imputers")
    def _validate_and_filter_imputers(raw_imputers: Dict[str, Union[str, Dict[str, Any]]],
                                      schema: StructType) -> Dict[str, Imputer]:
        imputers = {}
        for col, raw_imputer in raw_imputers.items():
            if isinstance(raw_imputer, str):
                raw_imputer = {"strategy": raw_imputer}

            if "strategy" not in raw_imputer:
                raise InvalidArgumentError(
                    f"Invalid imputation {raw_imputer} for column {col}. "
                    "Specify imputation method as a string or as a dictionary with key 'strategy'.")

            spark_type = schema[col].dataType
            fill_value = raw_imputer.get("fill_value", None)
            imputer = Imputer.create_imputer(raw_imputer["strategy"], col, spark_type, fill_value)
            if imputer:
                imputers[col] = imputer
        return imputers

    @chain_invalid_argument_error("primary_metric")
    def _validate_and_parse_metric(self, metric: Optional[Union[Metric, str]]) -> Metric:
        if metric is None:
            metric = self.default_metric

        supported_metrics = [metric.short_name for metric in self.supported_metrics()]
        if isinstance(metric, str):
            try:
                metric = Metric.get_metric(metric)
            except ValueError:
                raise InvalidArgumentError(f"Invalid metric name \"{metric}\". "
                                           f"Supported metric names are: {supported_metrics}")

        if metric not in self.supported_metrics():
            raise InvalidArgumentError(
                f"Provided metric is not among the supported metrics: {supported_metrics}")
        return metric

    def _validate_cluster(self) -> Dict[str, str]:
        runtime_version = self.spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion",
                                              None)
        isolation_enabled = self.spark.conf.get("spark.databricks.pyspark.enableProcessIsolation",
                                                None)
        table_acl_enabled = self.spark.conf.get("spark.databricks.acl.dfAclsEnabled", None)
        py4j_security_enabled = self.spark.conf.get("spark.databricks.pyspark.enablePy4JSecurity",
                                                    None)

        cluster_name = self.spark.conf.get("spark.databricks.clusterUsageTags.clusterName", None)
        cloud_provider = self.spark.conf.get("spark.databricks.cloudProvider", None)

        if runtime_version is not None and "-ml-" not in runtime_version:
            raise UnsupportedRuntimeError(
                "AutoML is only supported on the Databricks Runtime "
                "for Machine Learning. Please attach to a different cluster.")

        if isolation_enabled == "true":
            raise UnsupportedClusterError(
                "AutoML is currently not supported on clusters that enable process isolation eg: "
                "High Concurrency clusters with IAM Passthrough enabled. "
                "Please use a cluster with a different security configuration.")
        if table_acl_enabled == "true":
            raise UnsupportedClusterError(
                "AutoML is not supported on clusters that enable table ACLs. "
                "Please use a cluster with a different security configuration.")
        if py4j_security_enabled == "true":
            raise UnsupportedClusterError(
                "AutoML is not supported on clusters that enable py4j security. "
                "Please use a different cluster or set spark.databricks.pyspark.enablePy4JSecurity"
                " to false.")

        return {
            "runtime_version": runtime_version or "Unknown",
            "cluster_name": cluster_name or "Unknown",
            "cloud_provider": cloud_provider or "Unknown",
        }

    def _validate_params(
            self,
            timeout_minutes: Optional[int],
            horizon: Optional[int] = None,  # forecasting only
            frequency: Optional[str] = None,  # forecasting only
    ) -> None:
        # GUI should validate these before this point, because if these fail there is no experiment to set state
        if timeout_minutes is not None:
            if timeout_minutes < self.MINIMUM_TIMEOUT_MINUTES:
                raise InvalidArgumentError(
                    f"Timeout must be {self.MINIMUM_TIMEOUT_MINUTES} minutes or greater. Given value: {timeout_minutes}"
                )

    def _init_context(self, experiment_dir: Optional[str]) -> Context:
        session_id = str(uuid.uuid4())[:8]

        context = ContextFactory.get_context(
            context_type=self._context_type, session_id=session_id, experiment_dir=experiment_dir)
        return context

    @classmethod
    @abstractmethod
    def _get_supported_target_types(cls) -> List[DataType]:
        pass

    @abstractmethod
    def _get_planners(self, **kwargs) -> List[TrialPlanner]:
        pass

    def _save_and_init_cluster_envs(self) -> ClusterEnvironment:
        """
        Save and reset the AutoML related variables from users' running environment like spark confs and automl run.
        :return: saved environment values
        """
        # ML-13588: Disable hyperopt auto logging until we have a fix. Hyperopt autologging logs
        # MLflow to the default experiment of the notebook which will cause failures when we reset
        # the experiment in the notebook trial.
        hyperopt_conf = self.spark.conf.get("spark.databricks.mlflow.trackHyperopt.enabled", "true")
        self.spark.conf.set("spark.databricks.mlflow.trackHyperopt.enabled", "false")

        # End the mlflow run if there exists an active run
        active_runs = []
        while mlflow.active_run() is not None:
            mlflow_run_id = mlflow.active_run().info.run_id
            active_runs.append(mlflow_run_id)
            mlflow.end_run()

        return ClusterEnvironment(hyperopt_conf=hyperopt_conf, mlflow_run_ids=active_runs)

    def _restore_cluster_envs(self, user_env: ClusterEnvironment):
        # restore the conf for hyperopt autologging
        self.spark.conf.set("spark.databricks.mlflow.trackHyperopt.enabled", user_env.hyperopt_conf)

        # restore the mlflow run if exists
        for run in reversed(user_env.mlflow_run_ids):
            mlflow.start_run(run_id=run, nested=True)

    def _fit_run(
            self,
            dataset: Union[DataFrame, pd.DataFrame, ps.DataFrame, str],
            target_col: str,
            data_dir: Optional[str],
            exclude_cols: List[str],
            exclude_frameworks: Optional[List[str]],
            imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]],
            metric: Optional[Union[Metric, str]],
            max_trials: int,
            parallelism: int,
            timeout_minutes: Optional[int],
            experiment: Optional[mlflow.entities.Experiment],
            experiment_dir: Optional[str],
            time_col: Optional[str],
            run_id: str,  # set by @instrumented fit
            pos_label: Optional[ClassificationTargetTypes] = None,  # binary classification only
            identity_col: Optional[Union[str, list]] = None,  # forecasting only
            horizon: Optional[int] = None,  # forecasting only
            frequency: Optional[str] = None,  # forecasting only
            output_database: Optional[str] = None,  # forecasting only
            feature_store_lookups: Optional[List[Dict]] = None  # supervised learner only
    ) -> AutoMLSummary:
        cluster_env = self._save_and_init_cluster_envs()

        context = self._init_context(experiment_dir)

        # GUI should validate these before this point, because if these fail there is no experiment to set state
        self._validate_params(
            timeout_minutes=timeout_minutes,
            horizon=horizon,
            frequency=frequency,
        )

        metric = self._validate_and_parse_metric(metric)

        if isinstance(dataset, str):
            dataframe = self.spark.table(dataset)
            """
            If the user did not provide a string dataset name to the client, and instead provided
            a dataframe, then it would have been saved as a global temp view that we can now delete it.
            It is OK to delete the global temp view once the dataset is loaded using spark.table(dataset),
            and the dataframe can still be used after dropping the table.
            """
            shared_utils.drop_if_global_temp_view(dataset)
        else:
            temp_view_name = None
            dataframe = shared_utils.convert_to_spark_dataframe(dataset)
        if exclude_cols:
            exclude_cols = self._filter_exclude_cols(dataframe, exclude_cols, target_col, time_col)
            dataframe = dataframe.drop(*exclude_cols)
        else:
            exclude_cols = []
        if exclude_frameworks:
            exclude_frameworks = self._validate_and_filter_exclude_frameworks(exclude_frameworks)
        else:
            exclude_frameworks = set()
        if imputers:
            imputers = self._validate_and_filter_imputers(imputers, dataframe.schema)
        else:
            imputers = {}

        timeout = None
        if timeout_minutes is not None:
            timeout = timeout_minutes * 60

        context.create_or_set_experiment(experiment)

        # Anchor log that can be used to identify the streamGUID, clusterId and timeframe
        # for the this AutoML run using the experimentId and the problemType
        _logger.info(f"AutoML run for {self.problem_type.value} will record all trials to "
                     f"MlFlow experiment with id: {context.experiment_id}")

        alert_manager = AlertManager(context.experiment_id)

        try:
            context.set_experiment_init(
                target_col=target_col,
                data_dir=data_dir,
                timeout_minutes=timeout_minutes,
                max_trials=max_trials,
                problem_type=self.problem_type.value,
                evaluation_metric=metric)

            self._validate_column_uniqueness(dataframe, alert_manager)

            summary = self._fit_impl(
                dataset=dataframe,
                target_col=target_col,
                context=context,
                data_dir=data_dir,
                metric=metric,
                max_trials=max_trials,
                parallelism=parallelism,
                timeout=timeout,
                time_col=time_col,
                identity_col=identity_col,
                horizon=horizon,
                frequency=frequency,
                output_database=output_database,
                alert_manager=alert_manager,
                run_id=run_id,
                exclude_frameworks=exclude_frameworks,
                imputers=imputers,
                num_exclude_cols=len(exclude_cols),
                pos_label=pos_label,
                feature_store_lookups=feature_store_lookups)

            if shared_utils.is_automl_service_enabled() and shared_utils.use_automl_client():
                AutoMLSummary.save(summary)

        except KeyboardInterrupt:
            _logger.info(
                f"AutoML run with experiment id: {context.experiment_id} cancelled by the user.")
            context.set_experiment_state(RunState.CANCELED)
            raise
        except AutomlError as e:
            _logger.error(
                f"AutoML run with experiment id: {context.experiment_id} failed with {repr(e)}")
            context.set_experiment_state(RunState.FAILED)
            context.set_experiment_error(e.message)
            raise e
        except BaseException as e:
            _logger.error(
                f"AutoML run with experiment id: {context.experiment_id} failed with non-AutoML error {repr(e)}"
            )
            context.set_experiment_state(RunState.FAILED)
            context.set_experiment_error("An unknown error occurred")
            raise e
        else:
            _logger.info(f"AutoML run with experiment id: {context.experiment_id} succeeded.")
            context.set_experiment_state(RunState.SUCCESS)
        finally:
            self._restore_cluster_envs(cluster_env)

        return summary

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
            num_exclude_cols: int,  # only used for logging in supervised_learner.py
            identity_col: Optional[Union[str, list]] = None,  # forecasting only
            horizon: Optional[int] = None,  # forecasting only
            frequency: Optional[str] = None,  # forecasting only
            pos_label: Optional[ClassificationTargetTypes] = None,  # binary classification only
            feature_store_lookups: Optional[List[Dict]] = None  # supervised_learner only.
    ) -> AutoMLSummary:
        pass

    def _drop_invalid_rows(self, dataset: DataFrame, target_col: str, dataset_stats: InputStats,
                           alert_manager: AlertManager) -> DataFrame:
        if dataset_stats.num_target_nulls > 0:
            alert_manager.record(NullsInTargetColumnAlert(target_col))
            dataset = dataset.dropna(subset=[target_col])
        return dataset

    @staticmethod
    def _validate_dataset_has_rows(target_col: str, stats: InputStats,
                                   alert_manager: AlertManager) -> None:
        if stats.num_rows == 0:
            alert_manager.record(DatasetEmptyAlert())
            raise UnsupportedDataError("The input dataset is empty. Please pass in a valid dataset.")

        if stats.num_rows == stats.num_invalid_rows:
            alert_manager.record(AllRowsInvalidAlert(target_col))
            raise UnsupportedDataError(
                f"Every value in the selected target_col {target_col} is either null "
                "or does not have enough rows (5) per target class. Please pass in a valid dataset.")

    @staticmethod
    def _get_search_space(planners: List[TrialPlanner], max_trials: Optional[int] = None):
        model_search_spaces = [c.get_hyperparameter_search_space() for c in planners]
        search_space = hp.choice("model_type", model_search_spaces)
        return search_space

    @staticmethod
    def _get_post_cmd_instructions(experiment_url: str, metric: Metric) -> str:
        return textwrap.dedent(f"""
        **********************************************************************************************************
        Trials for training a model on the dataset have been kicked off. The model will be optimized
        for the {metric.description} metric (tracked as {metric.trial_metric_name} in MLflow experiments).
         
        You can track the completed trials in the MLflow experiment here:
        {experiment_url}

        Notebooks that generate the trials can be edited to tweak the setup, add hyperparameters and re-run the trials.
        All re-run notebooks will log the trials under the same experiment.
        Generated notebooks contain instructions to load models from your favorite trials.
        **********************************************************************************************************
        """)

    def _sort_trials(self, trials: List[TrialInfo], metric: Metric) -> List[TrialInfo]:
        return sorted(
            trials,
            key=lambda x: x.metrics.get(metric.trial_metric_name, metric.worst_value),
            reverse=metric.higher_is_better)

    def _clean_failed_mlflow_runs(self, experiment_id: str, data_source: DataSource,
                                  trials: List[TrialInfo]):
        """
        Clean up the failed MLflow runs.
        Since we have the retry logic. There might be some failed MLflow runs. We need to clean
        them up after all runs are finished.
        """
        trial_run_ids = {trial.mlflow_run_id for trial in trials}
        if not data_source.is_dbfs:
            trial_run_ids.add(data_source.run_id)
        exp_run_ids = {run.run_id for run in self._mlflow_client.list_run_infos(experiment_id)}

        run_ids_to_delete = exp_run_ids.difference(trial_run_ids)
        if run_ids_to_delete:
            _logger.debug(f"Deleting failed MlFlow runs with ids: {run_ids_to_delete}")
            for run_id in run_ids_to_delete:
                self._mlflow_client.delete_run(run_id)
