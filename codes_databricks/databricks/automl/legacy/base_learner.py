import logging
import textwrap
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable, Union

import mlflow
import pandas as pd
import pyspark.pandas as ps
from hyperopt import hp
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DataType, StructType

from databricks.automl.legacy.alerts import DuplicateColumnNamesAlert, NullsInTargetColumnAlert
from databricks.automl.legacy.alerts.alert_manager import AlertManager
from databricks.automl.legacy.const import ContextType, DatasetFormat, Framework, RunState
from databricks.automl.legacy.context import ContextFactory, Context, DataSource
from databricks.automl.legacy.errors import AutomlError, InvalidArgumentError, \
    UnsupportedDataError, UnsupportedRuntimeError, UnsupportedClusterError
from databricks.automl.legacy.imputers import Imputer
from databricks.automl.legacy.planner import TrialPlanner
from databricks.automl.legacy.problem_type import ClassificationTargetTypes, ProblemType, Metric
from databricks.automl.legacy.result import AutoMLSummary, TrialInfo
from databricks.automl.legacy.stats import PreSamplingStats

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
    SHOWING_IN_SCIENTIFIC_NOTATION_THRESHOLD_UPPER = 1e+6
    SHOWING_IN_SCIENTIFIC_NOTATION_THRESHOLD_LOWER = 1e-6

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

    def _convert_to_spark_dataframe(self, dataset: Union[DataFrame, pd.DataFrame, ps.DataFrame]
                                    ) -> Tuple[DataFrame, DatasetFormat]:
        """
        Validates the input dataset and returns a converted
        spark dataframe if the input is a pandas dataframe
        :param dataset: Either a Spark or a pandas DataFrame
        :return: Spark DataFrame
        """
        if isinstance(dataset, DataFrame):
            return dataset, DatasetFormat.SPARK
        if isinstance(dataset, pd.DataFrame):
            dataset = self.spark.createDataFrame(dataset)
            return dataset, DatasetFormat.PANDAS
        if isinstance(dataset, ps.DataFrame):
            dataset = dataset.to_spark()
            return dataset, DatasetFormat.PYSPARK_PANDAS
        raise UnsupportedDataError(
            f"input dataset is not a pyspark DataFrame, pyspark.pandas DataFrame or a pandas DataFrame: {type(dataset)}"
        )

    @staticmethod
    @chain_invalid_argument_error("exclude_columns")
    def _filter_exclude_cols(dataset: DataFrame, exclude_cols: List[str], target_col: str,
                             time_col: Optional[str]) -> Set[str]:
        """
        Validates the list of excluded columns provided by the user, and returns a sanitized list.
        Do not need to check if all columns get excluded. BaseDataPreprocessor._get_schema_info
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
            dataset: Union[DataFrame, pd.DataFrame, ps.DataFrame],
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
    ):
        cluster_env = self._save_and_init_cluster_envs()

        context = self._init_context(experiment_dir)

        # GUI should validate these before this point, because if these fail there is no experiment to set state
        self._validate_params(
            timeout_minutes=timeout_minutes,
            horizon=horizon,
            frequency=frequency,
        )

        metric = self._validate_and_parse_metric(metric)
        dataset, dataset_format = self._convert_to_spark_dataframe(
            dataset)  # returns a Spark dataframe
        if exclude_cols:
            exclude_cols = self._filter_exclude_cols(dataset, exclude_cols, target_col, time_col)
            dataset = dataset.drop(*exclude_cols)
        else:
            exclude_cols = []
        if exclude_frameworks:
            exclude_frameworks = self._validate_and_filter_exclude_frameworks(exclude_frameworks)
        else:
            exclude_frameworks = set()
        if imputers:
            imputers = self._validate_and_filter_imputers(imputers, dataset.schema)
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

            self._validate_column_uniqueness(dataset, alert_manager)

            summary = self._fit_impl(
                dataset=dataset,
                target_col=target_col,
                context=context,
                data_dir=data_dir,
                metric=metric,
                max_trials=max_trials,
                parallelism=parallelism,
                timeout=timeout,
                dataset_format=dataset_format,
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
                pos_label=pos_label)
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
            return summary
        finally:
            self._restore_cluster_envs(cluster_env)

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
            num_exclude_cols: int,  # only used for logging in supervised_learner.py
            identity_col: Optional[Union[str, list]] = None,  # forecasting only
            horizon: Optional[int] = None,  # forecasting only
            frequency: Optional[str] = None,  # forecasting only
            pos_label: Optional[ClassificationTargetTypes] = None,  # binary classification only
    ) -> AutoMLSummary:
        pass

    def _drop_invalid_rows(self, dataset: DataFrame, target_col: str,
                           dataset_stats: PreSamplingStats,
                           alert_manager: AlertManager) -> DataFrame:
        if dataset_stats.num_target_nulls > 0:
            alert_manager.record(NullsInTargetColumnAlert(target_col))
            dataset = dataset.dropna(subset=[target_col])
        return dataset

    @staticmethod
    def _get_search_space(planners: List[TrialPlanner], max_trials: Optional[int] = None):
        model_search_spaces = [c.get_hyperparameter_search_space() for c in planners]
        search_space = hp.choice("model_type", model_search_spaces)
        return search_space

    @classmethod
    def _get_display_format(cls, metric_value: float) -> str:
        abs_value = abs(metric_value)
        if (abs_value > cls.SHOWING_IN_SCIENTIFIC_NOTATION_THRESHOLD_UPPER) \
                or (abs_value < cls.SHOWING_IN_SCIENTIFIC_NOTATION_THRESHOLD_LOWER):
            return f"{metric_value:.6e}"
        else:
            return f"{metric_value:.3f}"

    @classmethod
    def _get_metrics_to_display(cls, metric: Metric, metrics: Dict[str, float]
                                ) -> List[Tuple[str, Tuple[str, str, str]]]:
        """
        Returns a dictionary with the key as metric name (without prefix) and value as a tuple of
        strings with (validation, train) metric values for the metric name.
        We also round the metric to 3 decimal points and return None if any of the metrics aren't logged
        :param metrics: metric dictionary logged into MLflow
        :return: {metric_name: (train_metrics, val_metrics, test_metrics)}
        """
        train_metrics = {}
        val_metric = {}
        test_metric = {}

        for metric_name, value in metrics.items():
            if metric_name.startswith("training_"):
                train_metrics[metric_name.replace("training_", "")] = cls._get_display_format(value)
            elif metric_name.startswith("val_"):
                val_metric[metric_name.replace("val_", "")] = cls._get_display_format(value)
            elif metric_name.startswith("test_"):
                test_metric[metric_name.replace("test_", "")] = cls._get_display_format(value)

        display_metrics = {}
        for metric_name in set(train_metrics.keys()).union(set(val_metric.keys())).union(
                set(test_metric.keys())):
            train = train_metrics.get(metric_name, "None")
            val = val_metric.get(metric_name, "None")
            test = test_metric.get(metric_name, "None")
            display_metrics[metric_name] = (train, val, test)

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
                <td> {trn} </td>
                <td> {val} </td>
                <td> {tst} </td>
            </tr>
            """
            for metric_name, (trn, val, tst) in cls._get_metrics_to_display(metric, trial.metrics)
        ]
        rows = "\n".join(formatted_rows)

        return f"""
                <table class="dataframe">
                    <thead>
                      <tr>
                        <th></th>
                        <th>Train</th>
                        <th>Validation</th>
                        <th>Test</th>
                      </tr>
                    </thead>
                    <tbody>
                    {rows}
                    </tbody>
                </table>
        """

    def _get_summary_html(self,
                          trial: TrialInfo,
                          data_exp_url: str,
                          experiment_url: str,
                          metric: Metric,
                          sample_fraction: Optional[float] = None) -> str:

        metric_table_html = self._get_metric_table(metric, trial)
        mlflow_exp_link = self._get_link_html(experiment_url, "MLflow experiment")
        data_exp_link = self._get_link_html(data_exp_url, "data exploration notebook")
        data_exp_div = f"<div><p>For exploratory data analysis, open the {data_exp_link}</p></div>"
        best_trial_notebook_link = self._get_link_html(trial.notebook_url, "best trial notebook")

        sampling_div = ""
        # NOTE: We don't do any sampling for ProblemType.FORECAST
        if sample_fraction and self.problem_type in {
                ProblemType.CLASSIFICATION, ProblemType.REGRESSION
        }:
            pct = sample_fraction * 100
            sampling_type = "stratified" if self.problem_type == ProblemType.CLASSIFICATION else "simple random"

            sampling_div = "<div><p><strong>NOTE:</strong> Data exploration and trials were run on a <strong>{:.3f}%</strong> sample of the usable rows in the dataset. This dataset was sampled using {} sampling.</p></div>".format(
                pct, sampling_type)

        return f"""
        <style>
            .grid-container {{
              display: grid
              grid-template-columns: auto;
              padding: 10px;
            }}
            <!-- Picked to be same as https://github.com/databricks/universe/blob/feaafc3875d9b95a124ed44ff4b99fb1002e544d/webapp/web/js/templates/iframeSandbox.css#L6-L11 -->
            .grid-container div {{
              font-family: Helvetica, Arial, sans-serif;
              font-size: 14px;
            }}
        </style>
        <div class="grid-container">
            {sampling_div}
            {data_exp_div}
            <div><p>To view the best performing model, open the {best_trial_notebook_link}</p></div>
            <div><p>To view details about all trials, navigate to the {mlflow_exp_link}</p></div>
            <div><p><strong>Metrics for the best trial:</strong></p></div>
            <div>
                <!-- class inlined from https://github.com/databricks/universe/blob/feaafc3875d9b95a124ed44ff4b99fb1002e544d/webapp/web/js/templates/iframeSandbox.css#L35 -->
                {metric_table_html}
            </div>
        </div>
        """

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

    @staticmethod
    def _get_link_html(url: str, text: str) -> str:
        return f"<a href={url}>{text}</a>"

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
