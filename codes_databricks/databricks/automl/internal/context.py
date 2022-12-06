from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from random import randint
from typing import Dict, Any, List, Optional, Tuple

import IPython.core.display as ipy
import mlflow
import mlflow.entities
import nbformat
from mlflow.tracking import MlflowClient
from nbclient.exceptions import DeadKernelError
from nbconvert.preprocessors import ExecutePreprocessor
from pyspark.sql import DataFrame

from databricks.automl.shared.errors import FeatureStoreError, InvalidArgumentError
from databricks.automl.internal.clients.jobs import JobsClient
from databricks.automl.internal.clients.workspace import WorkspaceClient
from databricks.automl.internal.common.const import ContextType, DatasetFormat, RunState
from databricks.automl.internal.errors import ExecutionTimeoutError, \
    ExperimentInitializationError
from databricks.automl.internal.errors import NotebookImportSizeExceededError, TrialFailedError, \
    UnsupportedRuntimeError, ExperimentDirectoryDoesNotExist
from databricks.automl.shared.errors import UnsupportedDataError
from databricks.automl.internal.plan import Plan
from databricks.automl.internal.sections.template import SectionTemplateManager
from databricks.automl.internal.sections.training.databricks import DatabricksPredefined
from databricks.automl.internal.sections.training.jupyter import JupyterDefinitions
from databricks.automl.shared.const import Metric, MLFlowFlavor
from databricks.automl.shared.databricks_utils import DatabricksUtils
from databricks.automl.shared.result import TrialInfo
from databricks.automl.shared.tags import Tag

from databricks.feature_store.entities.feature_spec import FeatureSpec

_logger = logging.getLogger(__name__)


class DataSource:
    def __init__(self,
                 dbfs_path: Optional[str] = None,
                 run_id: Optional[str] = None,
                 has_prefix: bool = False):
        """
        Data Source is wrapper class that stores information about where the user's training
        data is stored. It could be stored under a DBFS path passed by the user or as an MLflow
        artifact under an MLflow run.

        The data source requires that exactly one of the two parameters are present and the other
        one is null. If that's not the case creation of this class will fail the assertion.

        :param dbfs_path: Optional dbfs path where the data is stored
        :param run_id:    Optional run id of the MLflow run where the training data is logged as an artifact
        :param has_prefix: True if the data source requires a "file://" prefix
        """
        self._dbfs_path = dbfs_path
        self._run_id = run_id
        self._has_prefix = has_prefix

        assert (dbfs_path is None) ^ (run_id is None)
        "Either input_path or run_id should be set, not both"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

    @property
    def is_dbfs(self) -> bool:
        return self._dbfs_path is not None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def dbfs_path(self) -> str:
        return f"{self.file_prefix}{self._dbfs_path}"

    @property
    def dbfs_relative_path(self) -> str:
        return self._dbfs_path[5:]

    @property
    def file_prefix(self) -> str:
        return "file://" if self._has_prefix else ""


@dataclass
class NotebookDetails:
    id: int
    url: str
    path: str


class Context(ABC):
    """
    Abstract class for for AutoML execution context.
    """

    template_manager = SectionTemplateManager()

    @abstractmethod
    def __init__(self, session_id: str, experiment_dir: str, driver_notebook_name: str,
                 driver_notebook_url: str, notebook_file_ext: str):
        """
        :param session_id:            Session ID that's generated for each run of AutoML
        :param experiment_dir:        Base experiment directory where the generated experiment
                                      and notebooks and experiments are stored
        :param driver_notebook_name:  Name of the driver notebook which called AutoML
        :param driver_notebook_url:   URL of the driver notebook which called AutoML
        :param notebook_file_ext:     File extension to use when saving the generated notebooks
        """
        self._session_id = session_id
        self._experiment_dir = experiment_dir
        self._driver_notebook_name = driver_notebook_name
        self._driver_notebook_url = driver_notebook_url
        self._notebook_file_ext = notebook_file_ext

        self._date_str = self._get_current_datetime().strftime("%y-%m-%d-%H:%M")

        # setup required directories
        self._mkdir(self._experiment_dir)
        self._mkdir(self._trial_notebook_dir)
        self._mkdir(self._failed_notebook_dir)

        self._mlflow_client = MlflowClient()

        # experiment must be initialized by calling create_or_set_experiment
        self._experiment_id = None

    @staticmethod
    def _get_current_datetime() -> datetime:
        """
        For easier mocking in tests
        """
        return datetime.now()

    @property
    def _session_root_dir(self) -> str:
        """
        Parent directory for this AutoML session
        """
        session_dir = f"{self._date_str}-{self._driver_notebook_name}-{self._session_id}"
        return os.path.join(self._experiment_dir, session_dir)

    @property
    def _trial_notebook_dir(self) -> str:
        """
        Directory for saving the trial notebooks
        """
        return os.path.join(self._session_root_dir, "trials")

    @property
    def _failed_notebook_dir(self) -> str:
        """
        Directory to save the failed notebooks
        :return:
        """
        return os.path.join(self._session_root_dir, "failed-notebooks")

    def _get_data_storage_path(self, data_dir: str) -> str:
        """
        Validate and get the input data directory to store training data
        from the base data directory passed in by the user
        """
        if data_dir.startswith("dbfs:/"):
            data_dir = "/dbfs/" + data_dir[6:]
        if data_dir.startswith("file:///dbfs/"):
            data_dir = data_dir[7:]
        if data_dir.startswith("file:/dbfs/"):
            data_dir = data_dir[5:]
        if not data_dir.startswith("/dbfs/"):
            raise InvalidArgumentError("data_dir should start with `file:///dbfs/`, `file:/dbfs/` "
                                       f"`/dbfs/` or `dbfs:/`. Found data_dir={data_dir}")
        return os.path.join(data_dir, self._date_str, self._session_id)

    @property
    @abstractmethod
    def _has_data_source_prefix(self) -> bool:
        """
        Does the data source need a prefix for access
        """
        pass

    @staticmethod
    def _remove_worker_only_cells(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
        """
        Remove the notebook cells that are marked as worker only
        :param notebook: Jupyter notebook
        :return: Cleaned notebook
        """
        for cell in notebook.cells.copy():
            if cell.metadata.get("worker_config_only"):
                notebook.cells.remove(cell)
        return notebook

    @staticmethod
    def _remove_large_display_output_cells(
            notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
        """
        Remove the output cells for notebooks that have large display outputs
        and replace them with error output asking the user to re-run the notebooks
        :param notebook: Jupyter notebook
        :return: Notebook without the large display output
        """
        for cell in notebook.cells:
            if cell.metadata.get("large_display_output"):
                cell.outputs = [
                    nbformat.v4.new_output(
                        output_type="error",
                        ename="UnableToDisplay",
                        evalue="Please re-run this notebook to view the output of this cell")
                ]
        return notebook

    def save_training_data(
            self,
            dataset: DataFrame,
            data_dir: Optional[str],
            data_format: DatasetFormat,
            selected_cols: List[str],
            feature_spec: Optional[FeatureSpec] = None,
    ) -> Tuple[DataSource, DataFrame]:
        """
        Save the input training data to either DBFS or as an mlflow artifact

        :param dataset: Dataset to save
        :param data_dir: Optional data directory in DBFS where the data should be saved
        :param data_format: Dataset format to save
        :param selected_cols: Column names to be saved. If None, save all columns
        :param feature_spec: (Optional) feature_spec for feature store joins if user provided
        :return: DataSource of the saved training data, and the dataset with columns selected
        """

        if selected_cols:
            dataset = dataset.select(selected_cols)
        else:
            raise UnsupportedDataError("No columns are selected in dataset")

        if data_dir is not None:
            dbfs_path = self._get_data_storage_path(data_dir)
            os.makedirs(os.path.dirname(dbfs_path), exist_ok=True)

            data_source = DataSource(dbfs_path=dbfs_path, has_prefix=self._has_data_source_prefix)
            if data_format == DatasetFormat.PANDAS:
                dataset.toPandas().to_parquet(dbfs_path)
            elif data_format == DatasetFormat.SPARK:
                dataset.write.parquet(data_source.dbfs_path)
            else:
                raise UnsupportedDataError(f"saving dataset format is not a PySpark DataFrame, "
                                           f"or a pandas DataFrame: {data_format}")
            if feature_spec:
                feature_spec_file = os.path.join(dbfs_path, "feature_spec.yaml")
                try:
                    feature_spec.save(dbfs_path)
                except Exception as e:
                    raise FeatureStoreError(e)

            return data_source, dataset
        else:
            # create local temp dir to save dataset
            local_dir = self._local_dir_path
            session_dir = os.path.join(local_dir, "tmp", self._session_id)
            os.makedirs(session_dir, exist_ok=True)

            temp_file = os.path.join(session_dir, "training_data")

            # collect dataset and save to driver node
            if data_format == DatasetFormat.PANDAS:
                dataset.toPandas().to_parquet(temp_file)
            elif data_format == DatasetFormat.SPARK:
                dataset.toPandas().to_parquet(temp_file)
                # TODO: Fix the issue that saving parquet from spark to artifact does not work
                # tmp_data = DataSource(dbfs_path=temp_file, has_prefix=self._has_data_source_prefix)
                # dataset.write.parquet(tmp_data.dbfs_path)
            else:
                raise UnsupportedDataError(f"saving dataset format is not a PySpark DataFrame, "
                                           f"or a pandas DataFrame: {data_format}")

            if feature_spec:
                temp_feature_spec_file = os.path.join(session_dir, "feature_spec.yaml")
                try:
                    feature_spec.save(session_dir)
                except Exception as e:
                    raise FeatureStoreError(e)

            # generate a dummy run and log the saved data as an artifact
            with mlflow.start_run(
                    experiment_id=self.experiment_id,
                    run_name="Training Data Storage and Analysis") as run:
                mlflow.log_artifact(temp_file, artifact_path="data")
                if feature_spec:
                    mlflow.log_artifact(temp_feature_spec_file, artifact_path="data")
                run_id = run.info.run_id

            # delete the source of this run to avoid confusion
            existing_tags = mlflow.get_run(run_id=run_id).data.tags.keys()
            for tag in ["mlflow.source.name", "mlflow.source.type", "mlflow.databricks.notebookID"]:
                if tag in existing_tags:
                    self._mlflow_client.delete_tag(run_id, tag)

            # delete the temp data from the driver node
            shutil.rmtree(session_dir, True)

            return DataSource(run_id=run_id, has_prefix=self._has_data_source_prefix), dataset

    @property
    def driver_notebook_url(self) -> str:
        return self._driver_notebook_url

    @property
    @abstractmethod
    def _local_dir_path(self) -> str:
        """
        Temporary directory path on the local storage to store training data
        """
        pass

    @property
    def experiment_id(self):
        return self._experiment_id

    def get_experiment(self) -> mlflow.entities.Experiment:
        if not self._experiment_id:
            raise ExperimentInitializationError("Experiment must be initialized before usage.")
        return mlflow.get_experiment(self._experiment_id)

    @abstractmethod
    def get_experiment_url(self, absolute: bool = False) -> str:
        """
        Fetch the experiment url
        :param sort_metric: optional metric to sort the experiments by
        :param absolute: returns absolute url if true
        :return: Experiment Url
        """
        pass

    @abstractmethod
    def to_absolute_url(self, relative_url: str) -> str:
        """
        :param relative_url: relative URL
        :return: absolute URL, created from relative URL
        """
        pass

    @abstractmethod
    def display_html(self, html) -> None:
        """
        Displays the given html in the notebook
        :param html: html to display
        :return: HTML is displayed
        """
        pass

    @abstractmethod
    def _mkdir(self, dirname: str) -> None:
        """
        Makes the directory in the underlying storage

        :param dirname: Name of directory to create
        """
        pass

    @abstractmethod
    def _save_notebook(self,
                       notebook: nbformat.NotebookNode,
                       save_dir: str,
                       with_worker_only_cells: bool = False) -> NotebookDetails:
        """
        Saves the given notebook to the path that's provided for the underlying storage

        :param notebook: JupyterNotebook to save
        :param save_dir: Parent directory where the notebook should be saved
        :param with_worker_only_cells: Indecator to control whether to save the worker only cells
        :returns NotebookDetails with notebook_id, notebook_url and notebook_path
        """
        pass

    @abstractmethod
    def _pre_execution_setup(self, plan: Plan) -> Plan:
        """
        This is used to add additional cells that are required to run the notebooks in
        different environments (IPykernel & Databricks)
        """
        pass

    def _execute_notebook(
            self,
            plan: Plan,
            save_dir: str,
            timeout: Optional[int] = None,
            max_retries: int = 3,
            is_data_exploration: bool = False) -> Tuple[NotebookDetails, Dict[str, Any]]:
        """
        Execute the notebook plan using an IPython Kernel

        :param plan: Notebook Plan
        :param timeout: Timeout in seconds
        :param max_retries: Max number of times to retry a failed notebook
        :return: (Details about saved notebook, Map with results from the notebook execution)
        """
        # Different context will need different pre-execution setups
        plan = self._pre_execution_setup(plan)

        # Convert the plan to a jupyter notebook and execute it
        nb = plan.to_jupyter_notebook(experiment_id=self.experiment_id)
        result = None

        try:
            result, nb = self._execute_notebook_with_ipykernel(nb, timeout, max_retries)
        except Exception as e:
            if is_data_exploration:
                _logger.warning(f"Data exploration notebook failed with error {repr(e)}")
            else:
                raise e

        # Write the notebook
        details = self._save_notebook(nb, save_dir)
        return details, result

    def execute_data_exploration(self,
                                 plan: Plan,
                                 data_source: DataSource,
                                 timeout: Optional[int] = None,
                                 max_retries: int = 3) -> Tuple[NotebookDetails, Dict[str, Any]]:
        """
        Execute the data exploration notebook using IPython Kernel

        :param plan: Notebook Plan
        :param data_source: Data Source for the data
        :param timeout: Timeout in seconds
        :param max_retries: Max number of times to retry a failed notebook
        :return: (Notebook URL, Result of data exploration)
        """
        details, result = self._execute_notebook(
            plan,
            self._session_root_dir,
            timeout,
            max_retries,
            is_data_exploration=True,
        )
        if not data_source.is_dbfs:
            self._set_mlflow_run_source(
                run_id=data_source.run_id, notebook_name=plan.name, notebook_id=details.id)
            if not result:
                # result=None indicates that the data exploration notebook failed
                self._mlflow_client.set_terminated(data_source.run_id, status="FAILED")

        return details, result

    def execute_trial(self,
                      plan: Plan,
                      metric: Metric,
                      flavor: MLFlowFlavor,
                      timeout: Optional[int] = None,
                      max_retries: int = 3) -> TrialInfo:
        """
        Execute the trial plan using an IPython Kernel

        :param plan: Notebook Plan for the trial
        :param metric: Metric used to optimize the trials
        :param timeout: Timeout in seconds
        :param max_retries: Max number of times to retry a failed trial
        :param flavor: Mlflow flavor for the trial
        :return: TrialInfo for the executed trial
        """
        details, result = self._execute_notebook(plan, self._trial_notebook_dir, timeout,
                                                 max_retries)

        # Set MLflow source to the notebook with output
        run_id = result.get("mlflow_run_id")
        self._set_mlflow_run_source(run_id=run_id, notebook_name=plan.name, notebook_id=details.id)

        return TrialInfo(
            metric=metric,
            notebook_id=details.id,
            notebook_url=details.url,
            notebook_path=details.path,
            flavor=flavor,
            mlflow_run_id=result.get("mlflow_run_id", None))

    @abstractmethod
    def _execute_notebook_job(
            self,
            plan: Plan,
            save_dir: str,
            timeout: Optional[int] = None,
            max_retries: int = 3,
            is_data_exploration: bool = False) -> Tuple[NotebookDetails, Dict[str, Any]]:
        """
        Execute the notebook plan as a Databricks Job.
        NOTE: JupyterContext will still execute this notebook with IPython Kernel
        
        :param plan: Notebook Plan
        :param timeout: Timeout in seconds
        :return: (Details about saved notebook, Map with results from the notebook execution)
        """ ""
        pass

    def save_job_notebook(self, run_id: str, path: str) -> None:
        """
        Save the Databricks Job to notebook.
        NOTE: JupyterContext will do nothing
        
        :param run_id: Databricks run id
        :param path: Notebook path to save
        """ ""
        pass

    def save_job_notebooks(self, notebook_job_info: List[Dict[str, str]]):
        for job in notebook_job_info:
            if job["run_id"]:
                self.save_job_notebook(job["run_id"], job["path"])

    def execute_trial_job(self,
                          plan: Plan,
                          metric: Metric,
                          flavor: MLFlowFlavor,
                          timeout: Optional[int] = None,
                          max_retries: int = 3) -> TrialInfo:
        """
        Execute the trial plan as a Databricks Job.
        NOTE: JupyterContext will still execute this trial with IPython Kernel

        :param plan: Notebook Plan for the trial
        :param metric: Metric used to optimize the trials
        :param timeout: Timeout in seconds
        :param max_retries: Max number of times to retry a failed trial
        :param flavor: Mlflow flavor for the trial
        :return: TrialInfo for the executed trial
        """
        details, result = self._execute_notebook_job(plan, self._trial_notebook_dir, timeout,
                                                     max_retries)

        # Set MLflow source to the notebook with output
        run_id = result.get("mlflow_run_id")
        self._set_mlflow_run_source(run_id=run_id, notebook_name=plan.name, notebook_id=details.id)

        return TrialInfo(
            metric,
            notebook_id=details.id,
            notebook_url=details.url,
            notebook_path=details.path,
            flavor=flavor,
            mlflow_run_id=run_id,
            workflow_run_id=result.get("workflow_run_id"))

    def execute_data_exploration_job(
            self,
            plan: Plan,
            data_source: DataSource,
            timeout: Optional[int] = None,
            max_retries: int = 3) -> Tuple[NotebookDetails, Dict[str, Any]]:
        """
        Execute the data exploration notebook as a Databricks Job.
        NOTE: JupyterContext will still execute this trial with IPython Kernel

        :param plan: Notebook Plan
        :param data_source: Data Source for the data
        :param timeout: Timeout in seconds
        :param max_retries: Max number of times to retry a failed notebook
        :return: (Notebook URL, Result of data exploration)
        """
        details, result = self._execute_notebook_job(
            plan, self._session_root_dir, timeout, max_retries, is_data_exploration=True)
        if not data_source.is_dbfs:
            self._set_mlflow_run_source(
                run_id=data_source.run_id, notebook_name=plan.name, notebook_id=details.id)
            if not result:
                # result=None indicates that the data exploration notebook failed
                self._mlflow_client.set_terminated(data_source.run_id, status="FAILED")

        return details, result

    def _execute_notebook_with_ipykernel(self,
                                         nb: nbformat.NotebookNode,
                                         timeout: int,
                                         max_retries: int = 3
                                         ) -> Tuple[Dict[str, Any], nbformat.NotebookNode]:
        """
        Run the jupyter notebook with IPython Kernel and return the extracted results

        :param nb: Jupter notebook to run
        :param timeout: Timeout in seconds
        :param max_retries: maximum number of retries
        :return: result string of the last cell and the notebook with results
        """

        num_retries = 0
        while True:
            try:
                ep = ExecutePreprocessor(kernel_name="python3", timeout=timeout)
                ep.preprocess(nb)

                # fetch details about run from last cell
                result = json.loads(nb["cells"][-1].outputs[0]["text"])
                return result, nb
            except (Exception, DeadKernelError) as e:
                # Py4JJavaError is raised when nbformat.ExecutorPreprocessor.preprocess times out
                if "Cell execution timed out" in str(e):
                    _logger.debug("Notebook execution timed out.")
                    self._save_notebook(nb, self._failed_notebook_dir)
                    raise ExecutionTimeoutError()
                # Don't retry this trial if we failed because of OOM or if we have exhausted our
                # retries. Some hyperparameters are more susceptible to using more memory and
                # OOMing; so if a trial with a given set of hyperparameters OOMs then it's better to
                # not retry it and eat up more memory and instead let hyperopt pick some other
                # hyperparameters for the trial.
                if isinstance(e, DeadKernelError) or num_retries > max_retries:
                    if isinstance(e, DeadKernelError):
                        _logger.warning(f"Notebook execution OOMed with {repr(e)}")
                    details = self._save_notebook(nb, self._failed_notebook_dir)
                    raise TrialFailedError(
                        f"AutoML trial failed. For more details, check failed notebook at: \n{details.url}"
                    ) from e

                else:
                    num_retries += 1
                    cell_stderr_output = ""
                    for cell in nb["cells"]:
                        outputs = cell.get("outputs")
                        if not outputs:
                            continue
                        for output in outputs:
                            if output.get("name") == "stderr":
                                text = output.get("text")
                                if text:
                                    cell_stderr_output += "\n  * " + text
                    cell_output_str = "No cell stderr output"
                    if cell_stderr_output:
                        cell_output_str = "Cell stderr output:" + cell_stderr_output
                    _logger.warning(f"Retrying error: {repr(e)}\n{cell_output_str}")

    def create_or_set_experiment(self, experiment: Optional[mlflow.entities.Experiment]) -> None:
        """
        Create experiment or set to an existing one.

        :param experiment: MLflow experiment to initialize, or None to create a new one
        """
        if not experiment:
            exp_name = f"{self._driver_notebook_name}-Experiment-{self._session_id}"
            exp_path = os.path.join(self._session_root_dir, exp_name)
            self._experiment_id = mlflow.create_experiment(exp_path)
        else:
            self._experiment_id = experiment.experiment_id

    def set_experiment_state(self, state: RunState) -> None:
        """
        Update experiment state

        :param state: StateType indicating the current experiment state
        """
        self._mlflow_client.set_experiment_tag(
            experiment_id=self.experiment_id, key=Tag.STATE, value=state.name)

        # Note that start_time is recorded here, because it is not necessarily
        # the same as the creation time of the MLflow experiment or MLflow run.
        if state == state.RUNNING:
            self._mlflow_client.set_experiment_tag(
                experiment_id=self.experiment_id, key=Tag.START_TIME, value=int(time.time()))
        elif state in [state.CANCELED, state.FAILED, state.SUCCESS]:  # All terminal states
            self._mlflow_client.set_experiment_tag(
                experiment_id=self.experiment_id, key=Tag.END_TIME, value=int(time.time()))

    def set_experiment_init(self,
                            target_col: str,
                            data_dir: str,
                            timeout_minutes: Optional[int],
                            max_trials: int,
                            problem_type: str,
                            evaluation_metric: Metric,
                            job_run_id: Optional[int] = None) -> None:
        """
        Indicate experiment has started and is running

        :param target_col: Column name of target labels
        :param data_dir: DBFS path for intermediate data
        :param timeout_minutes: The maximum time to wait for the AutoML trials to complete
        :param max_trials: The maximum number of trials to run
        :param problem_type: classification, regression, forecasting, etc
        :param evaluation_metric: F1, etc
        :param job_run_id: run id of the job running AutoML
        """
        tag_map_d = {
            Tag.PROBLEM_TYPE: problem_type,
            Tag.TARGET_COL: target_col,
            Tag.DATA_DIR: data_dir,
            Tag.TIMEOUT_MINUTES: timeout_minutes,
            Tag.MAX_TRIALS: max_trials,
            Tag.EVALUATION_METRIC: evaluation_metric.trial_metric_name,
            Tag.EVALUATION_METRIC_ASC: not evaluation_metric.higher_is_better,
            Tag.JOB_RUN_ID: job_run_id,
        }

        # tags that should not be logged to mlflow if they are null
        remove_if_null_tags = [Tag.JOB_RUN_ID, Tag.TIMEOUT_MINUTES]
        for tag in remove_if_null_tags:
            if tag in tag_map_d and tag_map_d[tag] is None:
                tag_map_d.pop(tag)

        for key, val in tag_map_d.items():
            self._mlflow_client.set_experiment_tag(
                experiment_id=self.experiment_id, key=key, value=val)

        self._mlflow_client.set_experiment_tag(
            experiment_id=self.experiment_id, key=Tag.BASE, value=True)
        self.set_experiment_state(RunState.RUNNING)

    def set_experiment_error(self, msg):
        """
        Indicate the experiment had an error and quit

        :param msg: Human readable error message
        :return:
        """
        self._mlflow_client.set_experiment_tag(
            experiment_id=self.experiment_id, key=Tag.ERROR_MESSAGE, value=msg)

    def set_experiment_exploration_notebook(self, notebook_id) -> None:
        """
        Indicate the exploration notebook has finished running

        :param notebook_id: id of the data exploration notebook
        """
        self._mlflow_client.set_experiment_tag(
            experiment_id=self.experiment_id, key=Tag.EXPLORATION_NOTEBOOK_ID, value=notebook_id)

    def set_experiment_best_trial_notebook(self, notebook_id) -> None:
        """
        Indicate the best trial

        :param notebook_id: id of the best trial notebook
        """
        self._mlflow_client.set_experiment_tag(
            experiment_id=self.experiment_id, key=Tag.BEST_TRIAL_NOTEBOOK_ID, value=notebook_id)

    def set_sample_fraction(self, fraction: float) -> None:
        """
        Indicate the fraction to which the dataset is sampled to
        :param fraction: The fraction used to sample the dataset
        """
        self._mlflow_client.set_experiment_tag(
            experiment_id=self.experiment_id, key=Tag.SAMPLE_FRACTION, value=fraction)

    def _set_mlflow_run_source(self, run_id: str, notebook_name: str, notebook_id: int) -> None:
        """
        Sets the source of an mlflow run as the notebook id
        :param run_id: mlflow run id to set the source for
        :param notebook_name: name of notebook to use in source name
        :param notebook_id: databricks notebook id
        """
        existing_tags = mlflow.get_run(run_id=run_id).data.tags.keys()

        self._mlflow_client.set_tag(run_id, "mlflow.source.name", f"Notebook: {notebook_name}")
        self._mlflow_client.set_tag(run_id, "mlflow.source.type", "NOTEBOOK")
        self._mlflow_client.set_tag(run_id, "mlflow.databricks.notebookID", notebook_id)

        if "mlflow.databricks.notebookRevisionID" in existing_tags:
            self._mlflow_client.delete_tag(run_id, "mlflow.databricks.notebookRevisionID")

    def set_output_table_name(self, output_table_name) -> None:
        """
        Set the output table name for time series forecasting
        :param output_table_name: output table name
        """
        self._mlflow_client.set_experiment_tag(
            experiment_id=self.experiment_id, key=Tag.OUTPUT_TABLE_NAME, value=output_table_name)


class JupyterContext(Context):
    """
    Jupyter context for local execution (test only).
    """

    def __init__(self, session_id: str, experiment_dir: Optional[str]):
        super().__init__(
            session_id=session_id,
            experiment_dir=experiment_dir or tempfile.gettempdir(),
            driver_notebook_name="",
            driver_notebook_url="",
            notebook_file_ext=".ipynb")

    def get_experiment_url(self, absolute: bool = False) -> str:
        return f"{mlflow.get_tracking_uri()}/{self.experiment_id}"

    def to_absolute_url(self, relative_url: str) -> str:
        return relative_url

    def display_html(self, html) -> None:
        ipy.display(ipy.HTML(html))

    def _mkdir(self, dirname: str) -> None:
        os.makedirs(dirname, exist_ok=True)

    @property
    def _has_data_source_prefix(self) -> bool:
        return False

    @property
    def _local_dir_path(self) -> str:
        return tempfile.gettempdir()

    def _save_notebook(self,
                       notebook: nbformat.NotebookNode,
                       save_dir: str,
                       with_worker_only_cells: bool = False) -> NotebookDetails:
        # Clean up cells that shouldn't be saved
        if not with_worker_only_cells:
            notebook = self._remove_worker_only_cells(notebook)
        # extract the hashed name from the notebook
        name = f"{self._date_str}-{Plan.get_nb_name(notebook)}{self._notebook_file_ext}"
        path = os.path.join(save_dir, name)

        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

        # generate fake url to maintain structural parity with actual URL
        id = randint(1, 1000)
        url = f"#notebook/{id}"

        return NotebookDetails(id=id, url=url, path=path)

    def _pre_execution_setup(self, plan: Plan) -> Plan:
        plan = deepcopy(plan)
        predef_configs = JupyterDefinitions()
        plan.prepend(predef_configs)
        return plan

    def _execute_notebook_job(
            self,
            plan: Plan,
            save_dir: str,
            timeout: Optional[int] = None,
            max_retries: int = 3,
            is_data_exploration: bool = False) -> Tuple[NotebookDetails, Dict[str, Any]]:
        return self._execute_notebook(plan, save_dir, timeout)


# Note: This class is not fully covered by unit tests. Some functions are unit tested.
#       Rest of the functionality is tested using the AutoMLDustSuite
#       which runs this function on an actual cluster.
class DatabricksContext(Context):
    """
    Databricks context for executing notebooks on the cluster
    """
    _LOCAL_DIR_ENV_KEY = "SPARK_LOCAL_DIRS"

    def __init__(self, session_id: str, databricks_utils: DatabricksUtils,
                 experiment_dir: Optional[str]):
        self._databricks_utils = databricks_utils

        self._ws_client = WorkspaceClient(self._databricks_utils.api_url,
                                          self._databricks_utils.api_token)
        self._jobs_client = JobsClient(self._databricks_utils.api_url,
                                       self._databricks_utils.api_token)

        self._host_name = self._databricks_utils.host_name

        if self._databricks_utils.driver_notebook_path:
            self._driver_notebook_name = os.path.split(
                self._databricks_utils.driver_notebook_path)[1]
            self._driver_notebook_url = self._ws_client.get_notebook_url(
                self._databricks_utils.driver_notebook_path)
        else:
            self._driver_notebook_name = ""
            self._driver_notebook_url = ""

        experiment_dir = self._init_experiment_dir(experiment_dir, databricks_utils)

        super().__init__(
            session_id=session_id,
            experiment_dir=experiment_dir,
            driver_notebook_name=self._driver_notebook_name,
            driver_notebook_url=self._driver_notebook_url,
            notebook_file_ext="")

    def _init_experiment_dir(self, experiment_dir: Optional[str],
                             databricks_utils: DatabricksUtils) -> str:
        """
        Use the optionally passed experiment dir and validate it or use the default experiment_dir.
        This function has a little quirky behaviour as explained below:

        Normal API User --> valid non-home dir --> dir used as root dir for storing exp and notebooks
        Normal API User --> valid home dir --> home_dir/databricks_automl used as root dir
        Normal API User --> invalid dir --> throw InvalidArgumentError
        Normal API User --> None dir --> home_dir/databricks_automl used as root dir

        Normal GUI User --> valid home dir (passed by JS) --> home_dir/databricks_automl used as root dir
        DbAdmin GUI User --> valid home dir (passed by JS) --> /databricks_automl used as root dir
                                                               (since no default home_dir for DBAdmin)

        DBAdmin API User --> valid non-home dir --> dir used as root dir for storing exp and notebooks
        DBAdmin API User --> valid home dir --> /databricks_automl used as root dir (since no home_dir for DBAdmin)
        DBAdmin API User --> invalid dir --> throw InvalidArgumentError
        DBAdmin API User --> None dir --> /databricks_automl used as root dir (since no home_dir for DBAdmin)

        ST Jobs API User --> valid non-home dir --> dir used as root dir for storing exp and notebooks
        ST Jobs API User --> valid home dir --> home_dir/databricks_automl used as root dir
        ST Jobs API User --> invalid dir --> throw InvalidArgumentError
        ST Jobs API User --> None dir --> throw ExperimentDirectoryDoesNotExist
                                          (since default home_dir /Users/unknown doesn't exist)

        :param experiment_dir: Optional experiment_dir passed by user
        :param databricks_utils:   DBUtils Context Info
        :return:               Validated user passed experiment_dir or default experiment_dir or throw
                               appropriate exception
        """
        default_experiment_dir = os.path.join("/Users", self._databricks_utils.user)

        # validate the experiment_dir if it's passed in
        if experiment_dir is not None:
            # Override experiment_dir for DBAdmins if the experiment_dir matches their home dir
            # This happens when a DBAdmin tries to run AutoML through the UI
            if databricks_utils.is_user_dbadmin and \
                    os.path.normpath(experiment_dir) == os.path.normpath(default_experiment_dir):
                _logger.warning(
                    f"Identified a DBAdmin user running AutoML with experiment_dir=\"{experiment_dir}\" "
                    "which is their home directory. Overwriting experiment_dir to \"/\".")
                experiment_dir = "/databricks_automl"
            if not self._ws_client.exists(experiment_dir):
                raise InvalidArgumentError(
                    f"Experiment directory `{experiment_dir}` does not exist. Please pass a valid directory."
                )
        else:
            experiment_dir = default_experiment_dir

            # [ES-271371][ML-19787]: Username for users calling AutoML API from ST jobs is
            # incorrect and hence the corresponding home directory does not exist
            if not self._ws_client.exists(experiment_dir):
                raise ExperimentDirectoryDoesNotExist(
                    f"Home directory `{experiment_dir}` does not exist. "
                    f"Please pass a valid directory using experiment_dir=\"<dir-name>\" "
                    "param to override this default value.")

        # If the user passes their correct home directory or if we use the default,
        # create an additional folder under the directory to keep existing behaviour
        # and maintain better folder hygiene
        if os.path.normpath(experiment_dir) == os.path.normpath(default_experiment_dir):
            experiment_dir = os.path.join(experiment_dir, "databricks_automl")
        return experiment_dir

    def get_experiment_url(self, absolute: bool = False) -> str:
        return self._databricks_utils.get_experiment_url(self.experiment_id, absolute)

    def to_absolute_url(self, relative_url: str) -> str:
        return self._databricks_utils.to_absolute_url(relative_url)

    def display_html(self, html) -> None:
        self._databricks_utils.display_html(html)

    def _mkdir(self, dirname: str) -> None:
        self._ws_client.mkdirs(dirname)

    @property
    def _has_data_source_prefix(self) -> bool:
        return True

    @property
    def _local_dir_path(self) -> str:
        if self._LOCAL_DIR_ENV_KEY not in os.environ:
            raise UnsupportedRuntimeError(
                f"Environment variable \"{self._LOCAL_DIR_ENV_KEY}\" is not "
                "set on the driver node. Please set this variable before re-running AutoML")
        return os.environ[self._LOCAL_DIR_ENV_KEY]

    def _save_notebook(self,
                       notebook: nbformat.NotebookNode,
                       save_dir: str,
                       with_worker_only_cells: bool = False) -> NotebookDetails:
        # Clean up cells that shouldn't be saved
        if not with_worker_only_cells:
            notebook = self._remove_worker_only_cells(notebook)
        # extract the hashed name from the notebook
        name = f"{self._date_str}-{Plan.get_nb_name(notebook)}{self._notebook_file_ext}"
        path = os.path.join(save_dir, name)

        try:
            self._ws_client.import_nbformat(path, notebook)
        # if notebook is too big to import, try removing the large output cells and try again
        except NotebookImportSizeExceededError:
            notebook = self._remove_large_display_output_cells(notebook)
            self._ws_client.import_nbformat(path, notebook)

        id = self._ws_client.get_notebook_id(path)
        relative_url = self._ws_client.get_notebook_url(path)
        url = self.to_absolute_url(relative_url)

        return NotebookDetails(id=id, url=url, path=path)

    def _pre_execution_setup(self, plan: Plan) -> Plan:
        plan = deepcopy(plan)
        predef_configs = DatabricksPredefined(self._databricks_utils.api_url,
                                              self._databricks_utils.api_token)
        plan.prepend(predef_configs)
        return plan

    def _execute_notebook_job(
            self,
            plan: Plan,
            save_dir: str,
            timeout: Optional[int] = None,
            max_retries: int = 3,
            is_data_exploration: bool = False) -> Tuple[NotebookDetails, Dict[str, Any]]:
        # no extra setup is required when running the notebook as a Databricks job
        # so we just generate the notebook and save it
        nb = plan.to_jupyter_notebook(self.experiment_id)
        details = self._save_notebook(nb, save_dir, with_worker_only_cells=True)
        result_js = {}

        # run the notebook
        try:
            result = self._databricks_utils.run_notebook(details.path, timeout)
            result_js = json.loads(result)
            self.save_job_notebook(result_js.get("workflow_run_id"), details.path)
            return details, result_js
        except Exception as e:
            if is_data_exploration:
                _logger.warning(f"Data exploration notebook failed with error {repr(e)}")
                return details, result_js
            else:
                raise e

    def save_job_notebook(self, run_id: str, path: str) -> None:
        # export the run notebook from jobs and re-import it into workspace with output
        html = self._jobs_client.export(run_id)
        self._ws_client.import_html(path, html)

        # export the notebook again in nbformat re-import it
        # after removing the worker only cells
        notebook = self._ws_client.export_nbformat(path)
        notebook = self._remove_worker_only_cells(notebook)
        self._ws_client.import_nbformat(path, notebook)

    def set_experiment_init(self,
                            target_col: str,
                            data_dir: str,
                            timeout_minutes: Optional[int],
                            max_trials: int,
                            problem_type: str,
                            evaluation_metric: Metric,
                            job_run_id: Optional[int] = None):
        super().set_experiment_init(
            target_col,
            data_dir,
            timeout_minutes,
            max_trials,
            problem_type,
            evaluation_metric,
            job_run_id=self._databricks_utils.job_run_id)


class ContextFactory:
    @staticmethod
    def get_context(context_type: ContextType, session_id: str,
                    experiment_dir: Optional[str]) -> Context:
        """
        Generates contexts
        :return: Context
        :raises: UnsupportedRuntimeError if unsupported databricks runtime is selected
        """
        if context_type == ContextType.JUPYTER:
            return JupyterContext(session_id=session_id, experiment_dir=experiment_dir)
        elif context_type == ContextType.DATABRICKS:
            databricks_utils = DatabricksUtils.create()
            return DatabricksContext(
                session_id=session_id,
                databricks_utils=databricks_utils,
                experiment_dir=experiment_dir)
