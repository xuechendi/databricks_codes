import logging
import time
from typing import Union, Optional, List, Dict

import mlflow
import pandas as pd
import pyspark.pandas as ps
import pyspark.sql
from databricks.automl.client.protos.classification_pb2 import ClassificationParams
from databricks.automl.client.protos.common_pb2 import Experiment, FeatureStoreLookup
from databricks.automl.client.protos.forecasting_pb2 import ForecastingParams
from databricks.automl.client.protos.regression_pb2 import RegressionParams
from databricks.automl.client.protos.service_pb2 import CreateExperiment

from databricks.automl.client.service_client import AutomlServiceClient
from databricks.automl.client.validation import InputValidator, InputColumnParam
from databricks.automl.shared import utils as shared_utils
from databricks.automl.shared.cell_output import CellOutput
from databricks.automl.shared.const import ClassificationTargetTypes, Metric, ProblemType
from databricks.automl.shared.databricks_utils import DatabricksUtils
from databricks.automl.shared.result import AutoMLSummary
from databricks.automl.shared.tags import Tag

_logger = logging.getLogger(__name__)


class AutomlLifeCycleManager:
    def __init__(self):
        self._databricks_utils = DatabricksUtils.create()
        self._cluster_id = self._databricks_utils.cluster_id
        self._client = AutomlServiceClient(
            api_url=self._databricks_utils.api_url, api_token=self._databricks_utils.api_token)

    def execute_classification(
            self,
            dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame, str],
            target_col: str,
            primary_metric: str,
            time_col: Optional[str] = None,
            data_dir: Optional[str] = None,
            experiment_dir: Optional[str] = None,
            feature_store_lookups: Optional[List[Dict]] = None,
            exclude_cols: Optional[List[str]] = None,
            exclude_columns: Optional[List[str]] = None,
            exclude_frameworks: Optional[List[str]] = None,
            pos_label: Optional[ClassificationTargetTypes] = None,
            imputers: Optional[Dict[str, Union[str, Dict]]] = None,
            max_trials: Optional[int] = None,
            timeout_minutes: Optional[int] = None) -> Optional[AutoMLSummary]:

        InputValidator.warn_if_max_trials(max_trials)

        dataframe, table_name = InputValidator.get_dataframe_and_name(dataset)

        InputValidator.validate_cols_exists(
            schema=dataframe.schema,
            params=[
                InputColumnParam(name="target_col", input_cols=target_col, required=True),
                InputColumnParam(name="time_col", input_cols=time_col, required=False),
                InputColumnParam(
                    name="exclude_columns", input_cols=exclude_columns, required=False),
                InputColumnParam(name="exclude_cols", input_cols=exclude_cols, required=False),
            ])

        consolidated_exclude_params = InputValidator.consolidate_exclude_cols_params(
            exclude_columns=exclude_columns, exclude_cols=exclude_cols)

        imputers_proto = InputValidator.parse_imputers(imputers)

        feature_store_lookups_proto = self._create_feature_store_lookup_protos(feature_store_lookups)

        classification_params_proto = ClassificationParams(
            table_name=table_name,
            target_col=target_col,
            time_col=time_col,
            pos_label=str(pos_label) if pos_label else None,
            primary_metric=primary_metric,
            exclude_frameworks=exclude_frameworks,
            exclude_cols=consolidated_exclude_params,
            imputers=imputers_proto,
            feature_store_lookups=feature_store_lookups_proto)

        create_experiment_proto = CreateExperiment(
            existing_cluster_id=self._cluster_id,
            experiment_dir=experiment_dir,
            data_dir=data_dir,
            timeout_minutes=timeout_minutes,
            classification_params=classification_params_proto)

        return self._execute(create_experiment_proto, ProblemType.CLASSIFICATION, table_name,
                             primary_metric)

    def execute_regression(self,
                           dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame, str],
                           target_col: str,
                           primary_metric: str,
                           time_col: Optional[str] = None,
                           data_dir: Optional[str] = None,
                           experiment_dir: Optional[str] = None,
                           feature_store_lookups: Optional[List[Dict]] = None,
                           exclude_cols: Optional[List[str]] = None,
                           exclude_columns: Optional[List[str]] = None,
                           exclude_frameworks: Optional[List[str]] = None,
                           imputers: Optional[Dict[str, Union[str, Dict]]] = None,
                           max_trials: Optional[int] = None,
                           timeout_minutes: Optional[int] = None) -> Optional[AutoMLSummary]:

        InputValidator.warn_if_max_trials(max_trials)

        dataframe, table_name = InputValidator.get_dataframe_and_name(dataset)

        InputValidator.validate_cols_exists(
            schema=dataframe.schema,
            params=[
                InputColumnParam(name="target_col", input_cols=target_col, required=True),
                InputColumnParam(name="time_col", input_cols=time_col, required=False),
                InputColumnParam(
                    name="exclude_columns", input_cols=exclude_columns, required=False),
                InputColumnParam(name="exclude_cols", input_cols=exclude_cols, required=False),
            ])

        consolidated_exclude_params = InputValidator.consolidate_exclude_cols_params(
            exclude_columns=exclude_columns, exclude_cols=exclude_cols)

        imputers_proto = InputValidator.parse_imputers(imputers)

        feature_store_lookups_proto = self._create_feature_store_lookup_protos(feature_store_lookups)

        regression_params_proto = RegressionParams(
            table_name=table_name,
            target_col=target_col,
            time_col=time_col,
            primary_metric=primary_metric,
            exclude_frameworks=exclude_frameworks,
            exclude_cols=consolidated_exclude_params,
            imputers=imputers_proto,
            feature_store_lookups=feature_store_lookups_proto)

        create_experiment_proto = CreateExperiment(
            existing_cluster_id=self._cluster_id,
            experiment_dir=experiment_dir,
            data_dir=data_dir,
            timeout_minutes=timeout_minutes,
            regression_params=regression_params_proto)

        return self._execute(create_experiment_proto, ProblemType.REGRESSION, table_name,
                             primary_metric)

    def execute_forecasting(self,
                            dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame, str],
                            target_col: str,
                            time_col: str,
                            frequency: str,
                            horizon: int,
                            primary_metric: str,
                            identity_col: Optional[Union[str, List[str]]] = None,
                            data_dir: Optional[str] = None,
                            exclude_frameworks: Optional[List[str]] = None,
                            experiment_dir: Optional[str] = None,
                            max_trials: Optional[int] = None,
                            timeout_minutes: Optional[int] = None,
                            output_database: Optional[str] = None) -> Optional[AutoMLSummary]:

        InputValidator.warn_if_max_trials(max_trials)

        dataframe, table_name = InputValidator.get_dataframe_and_name(dataset)

        InputValidator.validate_cols_exists(
            schema=dataframe.schema,
            params=[
                InputColumnParam(name="target_col", input_cols=target_col, required=True),
                InputColumnParam(name="time_col", input_cols=time_col, required=True),
                InputColumnParam(name="identity_col", input_cols=identity_col, required=False),
            ])

        if isinstance(identity_col, str):
            identity_col = [identity_col]

        frequency_enum = InputValidator.parse_frequency(frequency)

        forecasting_params_proto = ForecastingParams(
            table_name=table_name,
            target_col=target_col,
            time_col=time_col,
            frequency=frequency_enum,
            horizon=horizon,
            identity_cols=identity_col,
            primary_metric=primary_metric,
            exclude_frameworks=exclude_frameworks,
            output_database=output_database)

        create_experiment_proto = CreateExperiment(
            existing_cluster_id=self._cluster_id,
            experiment_dir=experiment_dir,
            data_dir=data_dir,
            timeout_minutes=timeout_minutes,
            forecasting_params=forecasting_params_proto)

        return self._execute(create_experiment_proto, ProblemType.FORECAST, table_name,
                             primary_metric)

    def _execute(self, create_experiment_proto: CreateExperiment, problem_type: ProblemType,
                 table_name: str, primary_metric: str) -> Optional[AutoMLSummary]:
        """
        Calls the AutoML service to create an experiment and then periodically fetches
        results till the experiment reaches a terminal state.
        """
        metric = Metric.get_metric(primary_metric)
        _logger.info(
            f"AutoML will optimize for {metric.description} metric, which is tracked as {metric.trial_metric_name} in the MLflow experiment."
        )

        experiment_id = self._client.create_experiment(create_experiment_proto)
        experiment_url = self._databricks_utils.get_experiment_url(experiment_id, absolute=True)
        _logger.info(f"MLflow Experiment ID: {experiment_id}")
        _logger.info(f"MLflow Experiment: {experiment_url}")

        experiment = self._client.get_experiment(experiment_id)
        _logger.debug(f"Job run: {experiment.run_page_url}")

        trigger_mlflow_output = True
        log_data_exploration_url = True
        data_exp_notebook_url = None
        try:
            while True:
                try:
                    experiment = self._client.get_experiment(experiment_id)
                except KeyboardInterrupt:
                    # re-raise this if the user cancels when the control of execution is within the above try block
                    raise
                except Exception as ex:
                    # any other error we see should be caught, logged and continued
                    _logger.warning(f"Failed to fetch experiment info from AutoML service: {ex}")
                    time.sleep(10)
                    continue

                # When the mlflow experiment has been created, start a run to trigger the mlflow cell output
                if trigger_mlflow_output:
                    mlflow.start_run(experiment_id=experiment_id)
                    mlflow.end_run()
                    trigger_mlflow_output = False

                if log_data_exploration_url:
                    mlflow_experiment = mlflow.get_experiment(experiment_id)
                    if Tag.EXPLORATION_NOTEBOOK_ID in mlflow_experiment.tags:
                        data_exp_notebook_id = mlflow_experiment.tags.get(
                            Tag.EXPLORATION_NOTEBOOK_ID)
                        data_exp_notebook_url = self._databricks_utils.to_absolute_url(
                            f"#notebook/{data_exp_notebook_id}")
                        _logger.info(f"Data exploration notebook: {data_exp_notebook_url}")
                        log_data_exploration_url = False

                if experiment.state in {
                        Experiment.State.SUCCESS, Experiment.State.FAILED, Experiment.State.CANCELED
                }:
                    break
                else:
                    time.sleep(10)
        except KeyboardInterrupt:
            # catch ctrl-c and cancel the experiment for the user
            self._client.cancel_experiment(experiment_id)
            _logger.info(f"AutoML experiment is cancelled by the user")
            raise
        finally:
            # This block is always executed, even if the try block returns or an error is re-raised
            #
            # The temp view should've been deleted by the internal job, but if that job failed before deletion,
            # then the dataset is delete here. It is OK to call this multiple times, as the operation is idempotent
            shared_utils.drop_if_global_temp_view(table_name)

        return self._post_execute(problem_type, experiment.state, primary_metric, experiment_id,
                                  experiment_url, experiment.run_page_url, data_exp_notebook_url)

    def _post_execute(self, problem_type: ProblemType, experiment_state: Experiment.State,
                      primary_metric: str, experiment_id: str, experiment_url: str,
                      run_page_url: str, data_exp_notebook_url: str):
        """
        This is only called in _execute, but defined as a separate function for better unit testing
        """
        if experiment_state == Experiment.State.SUCCESS:
            _logger.info(f"AutoML experiment completed successfully.")
        elif experiment_state == Experiment.State.FAILED:
            _logger.info(
                f"AutoML experiment failed. Check the job logs for more information: {run_page_url}")
            return None
        elif experiment_state == Experiment.State.CANCELED:
            _logger.info(f"AutoML experiment is cancelled by the user")
            return None

        summary = AutoMLSummary.load(experiment_id)
        if summary:
            metric = Metric.get_metric(primary_metric)
            sample_fraction = summary.experiment.tags.get(Tag.SAMPLE_FRACTION, None)
            html_str = CellOutput.get_summary_html(summary.best_trial, data_exp_notebook_url,
                                                   experiment_url, metric, problem_type,
                                                   sample_fraction)
            self._databricks_utils.display_html(html_str)
        return summary

    @staticmethod
    def _create_feature_store_lookup_protos(feature_store_lookups: Optional[List[dict]]):
        if not feature_store_lookups:
            return []

        feature_store_lookup_protos = []
        for feature_store_lookup in feature_store_lookups:
            lookup_key = feature_store_lookup["lookup_key"]
            if isinstance(lookup_key, str):
                lookup_key = [lookup_key]
            feature_store_lookup_protos.append(
                FeatureStoreLookup(
                    table_name=feature_store_lookup["table_name"],
                    lookup_key=lookup_key,
                    timestamp_lookup_key=feature_store_lookup.get("timestamp_lookup_key")))

        return feature_store_lookup_protos
