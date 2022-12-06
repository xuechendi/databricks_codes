from abc import ABC
from typing import Any, Dict, List, Union

from databricks.automl.internal.alerts import StrongDatetimeSemanticTypeDetectionAlert, \
    StrongNumericSemanticTypeDetectionAlert, \
    StrongCategoricalSemanticTypeDetectionAlert, StrongTextSemanticTypeDetectionAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.alerts.feature_alert import *
from databricks.automl.shared.const import SemanticType


class FeatureAlertsHandler(ABC):
    """
    Helper class to parse feature alerts.
    """
    # Warning thresholds
    NULL_WARNING_THRESHOLD = 0.05

    # SemanticType to Alert Dict
    SEMANTIC_TYPE_DETECTION_ALERT_DICT = {
        SemanticType.DATETIME: StrongDatetimeSemanticTypeDetectionAlert,
        SemanticType.NUMERIC: StrongNumericSemanticTypeDetectionAlert,
        SemanticType.CATEGORICAL: StrongCategoricalSemanticTypeDetectionAlert,
        SemanticType.TEXT: StrongTextSemanticTypeDetectionAlert
    }

    def __init__(self, alert_manager: AlertManager):
        self._alert_manager = alert_manager

    def aggregate_warnings(self, warnings: Dict[str, Any]) -> List[FeatureAlert]:
        """
        Aggregate the warnings by warning types
        :param warnings: Warning message dictionary from pandas-profiling
        :return: the aggregated warning dictionary
        """
        warning_dict = {}
        for warning in warnings:
            # All pandas-profiling alerts: https://github.com/pandas-profiling/pandas-profiling/blob/develop/src/pandas_profiling/model/alerts.py#L14
            # pandas-profiling generates many alerts, but below are the only ones that automl need
            if warning["warning_type"] == "MISSING":
                if warning["fields"]["p_missing"] <= self.NULL_WARNING_THRESHOLD:
                    warning_class = SmallNullsColumnAlert
                else:
                    warning_class = LargeNullsColumnAlert
            elif warning["warning_type"] == "HIGH CORRELATION":
                warning_class = HighCorrelationColumnAlert
            elif warning["warning_type"] == "SKEWED":
                warning_class = SkewedColumnAlert
            elif warning["warning_type"] == "UNIFORM":
                warning_class = UniformColumnAlert
            elif warning["warning_type"] == "UNIQUE":
                warning_class = UniqueColumnAlert
            else:
                warning_class = None

            if warning_class:
                if warning_class in warning_dict.keys():
                    warning_dict[warning_class].add(warning["column_name"])
                else:
                    warning_dict[warning_class] = {warning["column_name"]}

        return [warning_class(cols) for warning_class, cols in warning_dict.items()]

    def log_warnings(self, warnings: List[FeatureAlert]) -> None:
        for warning in warnings:
            self._alert_manager.record(warning)

    def log_strong_semantic_detection_warnings(self,
                                               detections: Dict[SemanticType, List[str]]) -> None:
        for semantic_type, columns in detections.items():
            if columns and semantic_type in self.SEMANTIC_TYPE_DETECTION_ALERT_DICT:
                feature_alert = self.SEMANTIC_TYPE_DETECTION_ALERT_DICT[semantic_type]
                alert = feature_alert(sorted(columns))
                self._alert_manager.record(alert)
