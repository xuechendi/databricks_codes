from typing import Any, Dict, List, Optional, Tuple

from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.context import Context, DataSource, NotebookDetails
from databricks.automl.internal.plan import Plan
from databricks.automl.internal.sections.exploration.data import InputDataExplorationForecasting


class ForecastDataExplorationRunner:
    def __init__(self, context: Context, data_source: DataSource, target_col: str, time_col: str,
                 identity_col: Optional[List[str]], timeout: Optional[int], experiment_url: str,
                 cluster_info: Dict[str, str], alert_manager: AlertManager):
        self._alert_manager = alert_manager
        self._cluster_info = cluster_info
        self._context = context
        self._data_source = data_source
        self._experiment_url = experiment_url
        self._target_col = target_col
        self._time_col = time_col
        self._identity_col = identity_col
        self._timeout = timeout

    def run(self) -> Tuple[NotebookDetails, Dict[str, Any]]:
        # generate and run the data exploration notebook
        data_exploration_plan = Plan(
            name="DataExploration",
            sections=[
                InputDataExplorationForecasting(
                    data_source=self._data_source,
                    target_col=self._target_col,
                    time_col=self._time_col,
                    identity_col=self._identity_col,
                    experiment_url=self._experiment_url,
                    driver_notebook_url=self._context.driver_notebook_url,
                    cluster_info=self._cluster_info,
                    alerts=self._alert_manager.get_displayable_alerts(),
                )
            ])
        return self._context.execute_data_exploration_job(
            plan=data_exploration_plan, data_source=self._data_source, timeout=self._timeout)
