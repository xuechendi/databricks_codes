from __future__ import annotations  # see https://stackoverflow.com/questions/61544854/from-future-import-annotations

import logging
import os
import pickle
import traceback
from textwrap import dedent, indent
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
from mlflow.entities import Experiment

from databricks.automl.shared.const import Metric, MLFlowFlavor, SemanticType

_logger = logging.getLogger(__name__)


class TrialInfo:
    """
    Summary of an individual trial.

    Each trial includes metadata about the notebook, such as the URL and path.
    Additionally, MLflow run data is also exposed for visibility into model parameters, preprocessors, and
    training metrics.

    Example usage:
        >>> trial.notebook_id
        32466759
        >>> trial.notebook_url
        #notebook/32466759
        >>> trial.notebook_path
        /path/to/databricks_automl/notebooks/LogisticRegression
        >>> trial.duration
        7.019
        >>> trial.model
        LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class="auto", n_jobs=None, penalty="l2",
                   random_state=42,...
        >>> trial.metrics["val_f1_score"]
        0.8507936507936508

    To load the trained model and predict on new data:
        >>> model = trial.load_model()
        >>> model.predict(data)
        array([1, 0, 1])
    """

    def __init__(self,
                 metric: Metric,
                 notebook_id: int,
                 notebook_url: str,
                 notebook_path: str,
                 flavor: MLFlowFlavor,
                 mlflow_run_id: str,
                 workflow_run_id: Optional[str] = None):
        """
        Initializes TrialInfo with notebook metadata and MLflow run data.
        """
        self._metric = metric
        self._notebook_id = notebook_id
        self._notebook_url = notebook_url
        self._notebook_path = notebook_path
        self._model_flavor = flavor
        self._mlflow_run_id = mlflow_run_id
        self._workflow_run_id = workflow_run_id

        self._run = None

    def _get_run(self) -> int:
        if not self._run:
            self._run = mlflow.get_run(self._mlflow_run_id)
        return self._run

    @property
    def notebook_id(self) -> int:
        """Notebook id of the generated notebook"""
        return self._notebook_id

    @property
    def notebook_url(self) -> str:
        """Relative URL of the corresponding generated notebook."""
        return self._notebook_url

    @property
    def notebook_path(self) -> str:
        """Path to the corresponding generated notebook."""
        return self._notebook_path

    @property
    def mlflow_run_id(self) -> str:
        """MLflow run ID."""
        return self._mlflow_run_id

    @property
    def workflow_run_id(self) -> str:
        """Notebook job run ID if it is running as a job"""
        return self._workflow_run_id

    @property
    def metrics(self) -> Dict[str, float]:
        """Dictionary of training metrics for the trial."""
        return self._get_run().data.metrics

    @property
    def params(self) -> Dict[str, str]:
        """Dictionary of training parameters for the trial."""
        return self._get_run().data.params

    @property
    def model_path(self) -> str:
        """MLflow artifact URI of the trained model."""
        return os.path.join(self._get_run().info.artifact_uri, "model")

    @property
    def model_description(self) -> str:
        """Truncated description of the model and hyperparameters."""
        return self._get_run().data.params.get(
            "classifier",
            self._get_run().data.tags.get("estimator_name", "Unknown"))

    @property
    def duration(self) -> str:
        """Elapsed training time."""
        return "{:.3f} minutes".format(
            (self._get_run().info.end_time - self._get_run().info.start_time) / 60000.0)

    @property
    def preprocessors(self) -> str:
        """Description of preprocessors run before training."""
        return self._get_run().data.params.get("preprocessor__transformers", "None")

    @property
    def evaluation_metric_score(self) -> float:
        """Evaluation metric score of trained model."""
        return self.metrics.get(self._metric.trial_metric_name, self._metric.worst_value)

    def load_model(self) -> Any:
        """Loads the trained model."""
        if self._model_flavor == MLFlowFlavor.SKLEARN:
            return mlflow.sklearn.load_model(self.model_path)
        elif self._model_flavor in {MLFlowFlavor.PROPHET, MLFlowFlavor.ARIMA}:
            return mlflow.pyfunc.load_model(self.model_path)

    def __str__(self) -> str:
        return dedent(f"""
        Model: {self.model_description}
        Model path: {self.model_path}
        Preprocessors: {self.preprocessors}
        Training duration: {self.duration}
        Evaluation metric score: {self.evaluation_metric_score:.3f}
        Evaluation metric: {self._metric.description} (tracked as {self._metric.trial_metric_name})
        """)


class AutoMLSummary:
    """
    Summary of an AutoML run, including the MLflow experiment and list of detailed summaries for each trial.

    The MLflow experiment contains high level information, such as the root artifact location, experiment ID,
    and experiment tags. The list of trials contains detailed summaries of each trial, such as the notebook and model
    location, training parameters, and overall metrics.

    Example usage:
        >>> summary.experiment.experiment_id
        32639121
        >>> len(summary.trials)
        10
        >>> print(summary.best_trial)
        Model: DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion="gini",
                       max_depth=3, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                     ...
        Model path: dbfs:/databricks/mlflow-tracking/32639121/7ff5e517fd524f30a77b777f5be46d24/artifacts/model
        Preprocessors: [("onehot", OneHotEncoder(categories="auto", drop=None, dtype=<class "numpy.float64">,
                      handle_unknown="ignore", sparse=True), ["col2", "col3"])]
        Training duration: 0.056 minutes
        Evaluation metric: val_f1_score
        Evaluation metric score: 0.952
        >>> best_model = summary.best_trial.load_model()
        >>> best_model.predict(data)
        array([1, 0, 1])
    """

    def __init__(self,
                 experiment_id: str,
                 trials: List[TrialInfo],
                 semantic_type_conversions: Dict[SemanticType, List[str]] = {},
                 is_early_stopped: bool = False,
                 output_table_name: Optional[str] = None):
        """
        :param experiment_id: id of the MLflow experiment for AutoML run
        :param trials: List of TrialInfos for all trials, sorted descending by evaluation metric (best first)
        :param semantic_type_conversions: Dictionary of semantic type to columns detected and converted to that type
        :param is_early_stoppped: Whether the run is early stopped
        :param output_table_name: table name to save the prediction data from AutoML forecasting
        """
        self._experiment_id = experiment_id
        self._trials = trials
        self._semantic_type_conversions = {
            t.value: c
            for t, c in semantic_type_conversions.items() if c
        }
        self._is_early_stopped = is_early_stopped
        self._output_table_name = output_table_name

    def __str__(self) -> str:
        """
        Returns a string with a detailed summary of the best trial as well as statistics about the entire experiment.

        Example usage:
        >>> print(summary)
        Overall summary:
                Experiment ID: 32646004
                Number of trials: 10
                Evaluation metric distribution: min: 0.497, median: 0.612, max: 0.956
                Semantic type conversions: None
        Best trial:
                Model: DecisionTreeClassifier
                Model path: dbfs:/databricks/mlflow-tracking/32646004/3d6d726079a4439fb1bc687295f77da8/artifacts/model
                Preprocessors: None
                Training duration: 0.028 minutes
                Evaluation metric score: 0.952
        """
        best_trial_summary = indent(str(self.best_trial), "\t")

        return f"""Overall summary:
\tExperiment ID: {self._experiment_id}
\tNumber of trials: {len(self._trials)}
\tEvaluation metric distribution: {self.metric_distribution}
\tSemantic type conversions: {self._semantic_type_conversions if self._semantic_type_conversions else "None"}
Best trial:
{best_trial_summary}"""

    @property
    def experiment(self) -> Experiment:
        """The MLflow experiment object."""
        return mlflow.get_experiment(self._experiment_id)

    @property
    def trials(self) -> List[TrialInfo]:
        """The list of detailed summaries for each trial."""
        return self._trials

    @property
    def best_trial(self) -> TrialInfo:
        """The trial corresponding to the best performing model of all completed trials."""
        return self._trials[0]

    @property
    def metric_distribution(self) -> str:
        """The distribution of evaluation metric scores across trials."""
        n = len(self._trials)
        minimum = self._trials[-1].evaluation_metric_score
        maximum = self._trials[0].evaluation_metric_score
        median = np.median([trial.evaluation_metric_score for trial in self._trials])

        return "min: {min:.3f}, median: {med:.3f}, max: {max:.3f}".format(
            min=minimum, med=median, max=maximum)

    @property
    def semantic_type_conversions(self) -> Dict[str, List[str]]:
        """A dictionary of semantic type name to column names that AutoML detected and converted to that type."""
        return self._semantic_type_conversions

    @property
    def is_early_stopped(self) -> bool:
        """A boolean indicating whether the AutoML run is early stopped."""
        return self._is_early_stopped

    @property
    def output_table_name(self) -> str:
        """A string of the output table name for AutoML forecasting."""
        return self._output_table_name

    def __eq__(self, other) -> bool:
        return self._experiment_id == other._experiment_id

    @staticmethod
    def _get_summary_filepath(experiment_id: str) -> str:
        """
        Get the filepath to a temporary file on the cluster driver that would be used to save/load
        the AutoMLSummary object. Since there is a one-to-one relationship between automl
        experiments and mlflow experiments, we can use the experiment id as a unique filename.
        :return: a filepath to save the AutoMLSummary object
        """
        return f"/tmp/automl/summary_{experiment_id}"

    def _log_summary_error(action: str) -> None:
        _logger.error(
            f"AutoML raised a non-fatal exception. The AutoML experiment completed, but "
            f"encountered an error while {action} the AutoMLSummary object. Use the automl "
            f"experiment page to check results. Stack trace:\n{traceback.format_exc()}")

    @staticmethod
    def save(summary: AutoMLSummary) -> None:
        """
        Save the AutoMLSummary object to a file on the cluster driver using pickle.
        Creates directories if necessary.
        :return: filepath of the AutoMLSummary object
        """
        summary_filepath = AutoMLSummary._get_summary_filepath(summary.experiment.experiment_id)
        try:
            os.makedirs(os.path.dirname(summary_filepath), exist_ok=True)
            with open(summary_filepath, "wb") as summary_file:
                pickle.dump(summary, summary_file)
        except Exception as e:
            AutoMLSummary._log_summary_error(action="saving")

    @staticmethod
    def load(experiment_id: str) -> Optional[AutoMLSummary]:
        """
        Loads the AutoMLSummary object from a pickle file, then tries to delete the file.
        :return: the loaded AutoMLSummary object
        """
        summary_filepath = AutoMLSummary._get_summary_filepath(experiment_id)
        summary = None
        try:
            with open(summary_filepath, "rb") as summary_file:
                summary = pickle.load(summary_file)
        except Exception as e:
            AutoMLSummary._log_summary_error(action="loading")

        # Try to delete the file, but don't fail automl if deletion fails
        try:
            os.remove(summary_filepath)
        except Exception as e:
            _logger.error(
                f"AutoML raised a non-fatal exception. AutoML could not delete file "
                f"{summary_filepath}. AutoML no longer needs this file, and you may wish to "
                f"manually delete it. Stack trace:\n{traceback.format_exc()}")
        return summary
