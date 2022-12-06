import json
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Union

from mlflow.tracking import MlflowClient

from databricks.automl.legacy.alerts.alert import Alert


class AlertManager:
    """
    Responsible for recording alerts. At the moment, this is through mlflow's experiment tags, but it may change in the
    future. When record is called, the AlertManager immediately sets a tag on its associated experiment by setting the
    key as the alert name, and putting the rest of the information in the value as a JSON string.
    """

    def __init__(self, experiment_id: str):
        self._experiment_id = experiment_id
        self._mlflow_client = MlflowClient()
        self.alerts = []

    @staticmethod
    def _truncate(
            value: Union[Dict[str, Any], List[Any], str]) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns a possibly truncated copy of misc.
        Truncates each value to Alert.MAX_VALUE_LENGTH, and truncates each array to length Alert.MAX_VALUES_PER_KEY.
        """

        if isinstance(value, dict):
            truncated = deepcopy(value)
            for key, value in truncated.items():
                truncated[key] = AlertManager._truncate(value)
            return truncated
        elif isinstance(value, list):
            parsed_dict = {}
            new_value = []
            for i in range(min(len(value), Alert.MAX_VALUES_PER_KEY)):
                val_i = value[i]
                truncated_val = AlertManager._truncate(val_i)
                new_value.append(truncated_val)
            parsed_dict[Alert.VALUES] = new_value
            if len(value) > Alert.MAX_VALUES_PER_KEY:
                parsed_dict[Alert.OTHERS] = len(value) - Alert.MAX_VALUES_PER_KEY
            return parsed_dict
        elif isinstance(value, str):
            if len(value) > 2 * Alert.MAX_VALUE_LENGTH + 3:
                value = f"{value[:Alert.MAX_VALUE_LENGTH]}...{value[-Alert.MAX_VALUE_LENGTH:]}"
            return value

        return value

    def get_displayable_alerts(self) -> Dict[str, List[str]]:
        """
        Get all the alerts that can be displayed in the data exploration notebooks

        :return: {display_group: [alert_string_1, alert_string_2, ...]}
        """
        alerts = list(chain(*[alert.display() for alert in self.alerts if alert.display()]))
        displayable_alerts = defaultdict(list)
        for alert in alerts:
            displayable_alerts[alert.group.value].append(alert.string)
        return displayable_alerts

    def record(self, alert: Alert) -> None:
        """
        Given an alert, adds it the the _alerts array, then creates a key and value to set an MLflow experiment tag.
        """
        self.alerts.append(alert)
        key = alert.name
        value = {
            Alert.VERSION: alert.version,
            Alert.SEVERITY: alert.severity,
        }

        truncated_misc = AlertManager._truncate(alert.misc)
        value.update(truncated_misc)

        value.update(alert.get_additional_info())

        self._mlflow_client.set_experiment_tag(
            experiment_id=self._experiment_id, key=key, value=json.dumps(value))
