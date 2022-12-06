from typing import List, Dict, Optional

import nbformat

from databricks.automl.internal.sections.section import Section


class GlobalConfiguration(Section):
    """
    Section that contains global configurations.
    """
    _TITLE_TEMPLATE = "config.md.jinja"
    _CONFIG_TEMPLATE = "config.jinja"

    def __init__(self,
                 config_map: Dict[str, str],
                 model_name: str,
                 experiment_url: str,
                 notebook_url: str,
                 cluster_info: Dict[str, str],
                 name_prefix: str = "conf"):
        """
        :param model_name: name of the ML model which this section belongs to
        :param experiment_url: url of the MLflow experiment
        :param notebook_url: url of the original master notebook
        :param name_prefix: unique name prefix for internal variable and method names
        :param cluster_info: dictionary containing cluster metadata
        :param config_map: dictionary to specify global configurations, where values must be strings
        """
        self._name_prefix = name_prefix
        self._notebook_url = notebook_url
        self._experiment_url = experiment_url
        self._model_name = model_name
        self._cluster_info = cluster_info

        for v in config_map.values():
            assert isinstance(v, str), "Only string values are supported"
        self._config_map = config_map

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
        return list(self._config_map.keys())

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        title_cell = self.template_manager.render_markdown_cell(
            self._TITLE_TEMPLATE,
            model_name=self._model_name,
            driver_notebook_url=self._notebook_url,
            experiment_url=self._experiment_url,
            runtime_version=self._cluster_info.get("runtime_version"),
            cluster_name=self._cluster_info.get("cluster_name"))

        conf_cell = self.template_manager.render_code_cell(
            self._CONFIG_TEMPLATE, config=self._config_map)

        return [title_cell, conf_cell]


class ForecastGlobalConfiguration(GlobalConfiguration):
    """
        Section that contains global configurations for forecasting.
    """
    _CONFIG_TEMPLATE = "config_forecast.jinja"

    def __init__(self,
                 config_map: Dict[str, str],
                 identity_col: Optional[List[str]],
                 horizon: int,
                 var_id_cols: str,
                 var_horizon: str,
                 model_name: str,
                 experiment_url: str,
                 notebook_url: str,
                 cluster_info: Dict[str, str],
                 name_prefix: str = "conf"):
        super().__init__(config_map, model_name, experiment_url, notebook_url, cluster_info,
                         name_prefix)

        self._identity_col = identity_col
        self._multivariate = identity_col is not None
        self._horizon = horizon
        self._var_id_cols = var_id_cols
        self._var_horizon = var_horizon

    @property
    def output_names(self) -> List[str]:
        if self._multivariate:
            forecast_output = [self._var_id_cols, self._var_horizon]
        else:
            forecast_output = [self._var_horizon]
        return super().output_names + forecast_output

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        title_cell = self.template_manager.render_markdown_cell(
            self._TITLE_TEMPLATE,
            model_name=self._model_name,
            driver_notebook_url=self._notebook_url,
            experiment_url=self._experiment_url,
            runtime_version=self._cluster_info.get("runtime_version"),
            cluster_name=self._cluster_info.get("cluster_name"))

        conf_cell = self.template_manager.render_code_cell(
            self._CONFIG_TEMPLATE,
            identity_col=self._identity_col,
            multivariate=self._multivariate,
            horizon=self._horizon,
            var_id_cols=self._var_id_cols,
            var_horizon=self._var_horizon,
            config=self._config_map,
        )

        return [title_cell, conf_cell]
