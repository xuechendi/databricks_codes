from typing import Dict, List, Optional

import nbformat

from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.alerts.dataset_alert import DataExplorationTruncateRowsAlert, DataExplorationTruncateColumnsAlert
from databricks.automl.internal.common.const import CloudProvider, DatabricksDocumentationDomain
from databricks.automl.internal.context import DataSource
from databricks.automl.internal.sections.section import Section
from databricks.automl.shared.const import SemanticType, ProblemType


class InputDataExplorationSupervisedLearner(Section):
    """
    Section used to generate data exploration notebook
    """
    _NOTEBOOK_PANDAS_PROFILING_TEMPLATE = "data_exploration/pandas_profiling.notebook.jinja"
    _RETURN_PANDAS_PROFILING_RESULTS_TEMPLATE = "data_exploration/pandas_profiling.results.jinja"

    # Max number of cols to generate full pandas-profiling report. If there are too many
    # columns for the input dataset, we will generate a report with more expensive computations turned off.
    MAX_COLS_WITH_FULL_REPORT = 20

    # Max number of rows. If the dataframe has more columns, the notebook will truncate to first
    # MAX_ROWS rows before running pandas-profiling.
    MAX_ROWS = 10000

    # Max number of columns. If the dataframe has more columns, the notebook will select the
    # first MAX_COLS columns before running pandas-profiling.
    MAX_COLS = 100

    def __init__(
            self,
            data_source: DataSource,
            num_rows: int,
            num_cols: int,
            date_cols: List[str],
            numerical_cols: List[str],
            target_col: str,
            strong_semantic_detections: Dict[SemanticType, List[str]],
            experiment_url: str,
            driver_notebook_url: str,
            cluster_info: Dict[str, str],
            problem_type: ProblemType,
            alert_manager: AlertManager,
            extra_conf: Optional[str] = None,
            sample_fraction: Optional[float] = None,  # None indicates that no sampling was done
            time_col: Optional[str] = None,
            internal_cols: Optional[List[str]] = None,
            name_prefix: str = "exp"):
        self._data_source = data_source
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._date_cols = date_cols
        self._numerical_cols = numerical_cols
        self._target_col = target_col
        self._time_col = time_col
        self._internal_cols = internal_cols
        # convert key to str for ease of rendering in jinja, ignore keys pointing to empty lists
        self._strong_semantic_detections = {
            k.value: v
            for k, v in strong_semantic_detections.items() if v
        }
        self._experiment_url = experiment_url
        self._driver_notebook_url = driver_notebook_url
        self._cluster_info = cluster_info
        self._alert_manager = alert_manager
        self._sample_fraction = sample_fraction
        self._problem_type = problem_type.value
        self._extra_conf = extra_conf

        self._var_df = "df"
        self._var_features = "features"
        self._var_label = "label"
        self._var_memory_usage = "memory_usage"
        self._var_column_types = "column_types"
        self._var_categorical_stats = "categorical_stats"
        self._var_numeric_stats = "numeric_stats"
        self._var_null_stats_df = "null_stats_df"
        self._var_label_freq_df = "label_freq_df"
        self._var_profile = "df_profile"

        self._name_prefix = name_prefix

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def input_names(self) -> List[str]:
        return []

    @property
    def output_names(self) -> List[str]:
        return []

    def _generate_semantic_detection_doc_link(self) -> str:
        """
        Dynamically generates the documentation link for semantic type detection. We have a hard
        legal requirement on not showing AWS or GCP links in Azure workspaces, so we must generate
        the appropriate link based on the cloud provider set in the Spark conf.

        :return: The generated documentation link
        """
        doc_path = "/applications/machine-learning/automl.html#semantic-type-detection"
        cloud_provider = self._cluster_info.get("cloud_provider", None)

        if cloud_provider == CloudProvider.AZURE.value:
            # [SC-25822] For Azure, we remove all instances of ".html" from the URL.
            return DatabricksDocumentationDomain.AZURE.value + doc_path.replace(".html", "")
        elif cloud_provider == CloudProvider.GCP.value:
            return DatabricksDocumentationDomain.GCP.value + doc_path
        else:
            return DatabricksDocumentationDomain.AWS.value + doc_path

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        extra_conf = ""
        if self._extra_conf:
            extra_conf += self._extra_conf
        # Disable the interaction section if there's only a single continuous feature.
        if len(self._numerical_cols) == 1:
            extra_conf += " interactions={\"continuous\": False},"
        if self._num_cols > self.MAX_COLS_WITH_FULL_REPORT:
            extra_conf += " minimal=True,"

        # Non-constant low-cardinality numerical columns are treated as both numerical and
        # categorical during training. Since, in the current approach, each column can only be
        # treated as one or the other, treat such columns as numerical so that their interaction
        # graphs will be generated for them.
        strong_semantic_detections = self._strong_semantic_detections.copy()
        if SemanticType.CATEGORICAL.value in strong_semantic_detections:
            for numerical_col in self._numerical_cols:
                if numerical_col in strong_semantic_detections[SemanticType.CATEGORICAL.value]:
                    strong_semantic_detections[SemanticType.CATEGORICAL.value].remove(numerical_col)
            if not strong_semantic_detections[SemanticType.CATEGORICAL.value]:
                del strong_semantic_detections[SemanticType.CATEGORICAL.value]

        semantic_detection_doc_link = self._generate_semantic_detection_doc_link()

        nb_cells = self.template_manager.render_multicells(
            self._NOTEBOOK_PANDAS_PROFILING_TEMPLATE,
            var_df=self._var_df,
            var_profile=self._var_profile,
            extra_conf=extra_conf,
            load_from_dbfs=self._data_source.is_dbfs,
            dbfs_path=self._data_source.dbfs_path,
            data_run_id=self._data_source.run_id,
            date_cols=self._date_cols,
            target_col=self._target_col,
            time_col=self._time_col,
            strong_semantic_detections=strong_semantic_detections,
            experiment_url=self._experiment_url,
            driver_notebook_url=self._driver_notebook_url,
            runtime_version=self._cluster_info.get("runtime_version"),
            cluster_name=self._cluster_info.get("cluster_name"),
            sample_fraction=self._sample_fraction,
            problem_type=self._problem_type,
            alerts=self._alert_manager.get_displayable_alerts(),
            semantic_detection_doc_link=semantic_detection_doc_link,
            truncate_rows=self.MAX_ROWS if self._num_rows > self.MAX_ROWS else 0,
            truncate_cols=self.MAX_COLS if self._num_cols > self.MAX_COLS else 0,
            internal_cols=self._internal_cols,
        )
        if self._num_rows > self.MAX_ROWS:
            self._alert_manager.record(DataExplorationTruncateRowsAlert())
        if self._num_cols > self.MAX_COLS:
            self._alert_manager.record(DataExplorationTruncateColumnsAlert())
        exit_cells = self.template_manager.render_multicells(
            self._RETURN_PANDAS_PROFILING_RESULTS_TEMPLATE,
            metadata={"worker_config_only": True},
            var_df=self._var_df,
            var_profile=self._var_profile,
            target_col=self._target_col,
        )
        return nb_cells + exit_cells


class InputDataExplorationForecasting(Section):
    """
    Section used to generate data exploration notebook for timer series forecasting
    """
    _NOTEBOOK_TEMPLATE = "data_exploration/forecast.notebook.jinja"
    _RETURN_RESULTS_TEMPLATE = "data_exploration/forecast.results.jinja"

    def __init__(self,
                 data_source: DataSource,
                 target_col: str,
                 time_col: str,
                 identity_col: Optional[List[str]],
                 experiment_url: str,
                 driver_notebook_url: str,
                 cluster_info: Dict[str, str],
                 alerts: Dict[str, List[str]],
                 name_prefix: str = "exp"):
        self._data_source = data_source
        self._target_col = target_col
        self._time_col = time_col
        self._identity_col = identity_col
        self._multivariate = identity_col is not None
        self._experiment_url = experiment_url
        self._driver_notebook_url = driver_notebook_url
        self._cluster_info = cluster_info
        self._alerts = alerts

        self._var_df = "df"
        self._var_time_range = "df_time_range"
        self._var_label = "label"
        self._var_target_col = "target_col"
        self._var_memory_usage = "memory_usage"
        self._var_column_types = "column_types"
        self._var_categorical_stats = "categorical_stats"
        self._var_numeric_stats = "numeric_stats"
        self._var_target_stats_df = "target_stats_df"
        self._var_null_stats_df = "null_stats_df"
        self._var_label_freq_df = "label_freq_df"
        self._var_profile = "df_profile"

        self._name_prefix = name_prefix

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def input_names(self) -> List[str]:
        return []

    @property
    def output_names(self) -> List[str]:
        return []

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        nb_cells = self.template_manager.render_multicells(
            self._NOTEBOOK_TEMPLATE,
            prefix=self._name_prefix,
            var_df=self._var_df,
            var_target_col=self._var_target_col,
            var_time_range=self._var_time_range,
            var_null_stat_df=self._var_null_stats_df,
            var_target_stats_df=self._var_target_stats_df,
            load_from_dbfs=self._data_source.is_dbfs,
            dbfs_path=self._data_source.dbfs_path,
            data_run_id=self._data_source.run_id,
            file_prefix=self._data_source.file_prefix,
            target_col=self._target_col,
            time_col=self._time_col,
            multivariate=self._multivariate,
            identity_col=self._identity_col,
            experiment_url=self._experiment_url,
            driver_notebook_url=self._driver_notebook_url,
            runtime_version=self._cluster_info.get("runtime_version"),
            cluster_name=self._cluster_info.get("cluster_name"),
            alerts=self._alerts,
        )
        exit_cells = self.template_manager.render_multicells(
            self._RETURN_RESULTS_TEMPLATE,
            prefix=self._name_prefix,
            metadata={"worker_config_only": True},
            var_df=self._var_df,
            var_target_col=self._var_target_col,
            var_time_range=self._var_time_range,
            var_null_stat_df=self._var_null_stats_df,
            var_target_stats_df=self._var_target_stats_df,
            multivariate=self._multivariate,
            identity_col=self._identity_col,
            target_col=self._target_col,
        )
        return nb_cells + exit_cells
