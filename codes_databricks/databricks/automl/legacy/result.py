import os
from textwrap import dedent, indent
from typing import Any, Dict, List, Optional, Union

import mlflow
from mlflow.entities import Experiment
import numpy as np

from databricks.automl.legacy.const import MLFlowFlavor, SemanticType
from databricks.automl.legacy.forecast_preprocess import ForecastDataPreprocessResults
from databricks.automl.legacy.preprocess import SupervisedLearnerDataPreprocessResults
from databricks.automl.legacy.problem_type import Metric


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

        self._run = mlflow.get_run(self._mlflow_run_id)

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
        return self._run.data.metrics

    @property
    def params(self) -> Dict[str, str]:
        """Dictionary of training parameters for the trial."""
        return self._run.data.params

    @property
    def model_path(self) -> str:
        """MLflow artifact URI of the trained model."""
        return os.path.join(self._run.info.artifact_uri, "model")

    @property
    def model_description(self) -> str:
        """Truncated description of the model and hyperparameters."""
        return self._run.data.params.get("classifier",
                                         self._run.data.tags.get("estimator_name", "Unknown"))

    @property
    def duration(self) -> str:
        """Elapsed training time."""
        return "{:.3f} minutes".format(
            (self._run.info.end_time - self._run.info.start_time) / 60000.0)

    @property
    def preprocessors(self) -> str:
        """Description of preprocessors run before training."""
        return self._run.data.params.get("preprocessor__transformers", "None")

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
                 preprocess_result: Union[SupervisedLearnerDataPreprocessResults,
                                          ForecastDataPreprocessResults],
                 experiment: Experiment,
                 trials: List[TrialInfo],
                 semantic_type_conversions: Dict[SemanticType, List[str]] = {},
                 is_early_stopped: bool = False,
                 output_table_name: Optional[str] = None):
        """
        :param experiment: MLflow experiment object for AutoML run
        :param trials: List of TrialInfos for all trials, sorted descending by evaluation metric (best first)
        :param semantic_type_conversions: Dictionary of semantic type to columns detected and converted to that type
        :param is_early_stoppped: Whether the run is early stopped
        :param output_table_name: table name to save the prediction data from AutoML forecasting
        """
        self._preprocess_result = preprocess_result
        self._experiment = experiment
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
\tExperiment ID: {self._experiment.experiment_id}
\tNumber of trials: {len(self._trials)}
\tEvaluation metric distribution: {self.metric_distribution}
\tSemantic type conversions: {self._semantic_type_conversions if self._semantic_type_conversions else "None"}
Best trial:
{best_trial_summary}"""

    @property
    def preprocess_result(
            self) -> Union[SupervisedLearnerDataPreprocessResults, ForecastDataPreprocessResults]:
        return self._preprocess_result

    @property
    def experiment(self) -> Experiment:
        """The MLflow experiment object."""
        return self._experiment

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
