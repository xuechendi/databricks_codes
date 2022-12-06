from __future__ import annotations

from enum import Enum, unique, auto
from typing import Union

from pyspark.databricks.sql import annotation_utils

# this should be consistent with _get_supported_target_types in classifier.py
ClassificationTargetTypes = Union[int, bool, str]

# Global temporary views are stored in this database, defined by spark:
# https://spark.apache.org/docs/latest/sql-getting-started.html#global-temporary-view
GLOBAL_TEMP_DATABASE = "global_temp"


# The enum values must be strings because they are used to construct user-facing output
class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECAST = "forecast"
    TEST = "test"  # only used in unit tests


@unique
class SemanticType(Enum):
    """
    Detected ML feature type of a column.
    """
    DATETIME = annotation_utils.DATETIME
    NUMERIC = annotation_utils.NUMERIC
    CATEGORICAL = annotation_utils.CATEGORICAL
    NATIVE = annotation_utils.NATIVE
    TEXT = annotation_utils.TEXT


@unique
class MLFlowFlavor(Enum):
    SKLEARN = 1
    PROPHET = 2
    ARIMA = 3


@unique
class TimeSeriesFrequency(Enum):
    S = auto(), 1, "SECOND"
    sec = auto(), 1, "SECOND"
    second = auto(), 1, "SECOND"
    seconds = auto(), 1, "SECOND"
    T = auto(), 60, "MINUTE"
    m = auto(), 60, "MINUTE"
    min = auto(), 60, "MINUTE"
    minute = auto(), 60, "MINUTE"
    minutes = auto(), 60, "MINUTE"
    h = auto(), 3600, "HOUR"
    hr = auto(), 3600, "HOUR"
    hour = auto(), 3600, "HOUR"
    hours = auto(), 3600, "HOUR"
    D = auto(), 86400, "DAY"
    d = auto(), 86400, "DAY"
    day = auto(), 86400, "DAY"
    days = auto(), 86400, "DAY"
    W = auto(), 604800, "WEEK"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        obj._value_in_seconds = args[1]
        obj._proto_enum_value = args[2]
        return obj

    @property
    def value_in_seconds(self):
        return self._value_in_seconds

    @property
    def proto_enum_value(self):
        return self._proto_enum_value


# The information we capture about the metric.
class Metric(Enum):
    # Short name for the metric. Also surfaced in user-facing APIs.
    short_name: str
    # Metric name is it is recorded in the trials.
    trial_metric_name: str
    # Human-readable description for the metric. Surfaced in notebook cell outputs.
    description: str
    # Iff true then higher metric values are preferred.
    higher_is_better: bool

    # Enum definitions. The order of values in the tuple must follow the
    # ordering of arguments in the constructor. Unfortunately Python enums do not allow for named
    # arguments here.
    F1_SCORE = (
        "f1",  # short_name
        "val_f1_score",  # trial_metric_name
        "F1 score",  # description
        True,  # higher_is_better
    )
    LOG_LOSS = (
        "log_loss",  # short_name
        "val_log_loss",  # trial_metric_name
        "Log loss",  # description
        False,  # higher_is_better
    )
    PRECISION = (
        "precision",  # short_name
        "val_precision_score",  # trial_metric_name
        "Precision",  # description
        True,  # higher_is_better
    )
    ACCURACY = (
        "accuracy",  # short_name
        "val_accuracy_score",  # trial_metric_name
        "Accuracy",  # description
        True,  # higher_is_better
    )
    ROC_AUC = (
        "roc_auc",  # short_name
        "val_roc_auc_score",  # trial_metric_name
        "ROC/AUC",  # description
        True,  # higher_is_better
    )
    R2_SCORE = (
        "r2",  # short_name
        "val_r2_score",  # trial_metric_name
        "R2",  # description
        True,  # higher_is_better
    )
    MAE = (
        "mae",  # short_name
        "val_mae",  # trial_metric_name
        "mean absolute error",  # description
        False,  # higher_is_better
    )
    RMSE = (
        "rmse",  # short_name
        "val_rmse",  # trial_metric_name
        "root mean squared error",  # description
        False,  # higher_is_better
    )
    MSE = (
        "mse",  # short_name
        "val_mse",  # trial_metric_name
        "mean squared error",  # description
        False,  # higher_is_better
    )
    MAPE = (
        "mape",  # short_name
        "val_mape",  # trial_metric_name
        "mean absolute percentage error",
        False,
    )
    MDAPE = (
        "mdape",  # short_name
        "val_mdape",  # trial_metric_name
        "median absolute percentage error",  # description
        False,  # higher_is_better
    )
    SMAPE = (
        "smape",  # short_name
        "val_smape",  # trial_metric_name
        "symmetric mean absolute percentage error",  # description
        False,  # higher_is_better
    )
    COVERAGE = (
        "coverage",  # short_name
        "val_coverage",  # trial_metric_name
        "coverage of the upper and lower intervals",  # description
        True,  # higher_is_better
    )

    @classmethod
    def get_metric(cls, name: str) -> Metric:
        """
        Retrieves the metric matching a short-hand name.
        :param name: The shorthand name
        :return: The metric matching the shorthand. Raises ValueError if the name cannot be matched.
        """
        matches = [metric for metric in Metric if metric.short_name == name.lower()]
        if len(matches) != 1:
            raise ValueError(f"Cannot find metric with short name: {name}")
        return matches[0]

    def __init__(self, short_name: str, trial_metric_name: str, description: str,
                 higher_is_better: bool):
        self.short_name = short_name
        self.trial_metric_name = trial_metric_name
        self.description = description
        self.higher_is_better = higher_is_better

    def __eq__(self, other):
        return self.name == other.name and self.trial_metric_name == other.trial_metric_name and self.higher_is_better == other.higher_is_better

    @property
    def worst_value(self) -> float:
        return -float("inf") if self.higher_is_better else float("inf")
