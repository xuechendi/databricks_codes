from typing import Any, Dict, List

from databricks.automl.legacy.alerts.alert import Alert, Severity
from databricks.automl.legacy.problem_type import ClassificationTargetTypes
from databricks.automl.legacy.tags import Tag


class DatasetAlert(Alert):
    """
    Abstract class for alerts that apply to the entire dataset.
    """

    def __init__(self,
                 name: Tag,
                 severity: Severity,
                 version: int,
                 additional_info: Dict[str, Any] = None):
        super().__init__(
            name=name, severity=severity, version=version, additional_info=additional_info)


class CreateTableNotPermittedAlert(DatasetAlert):
    """
    Concrete alert class when user does not have permission to create tables
    """

    def __init__(self):
        super().__init__(
            name=Tag.ALERT_NO_PERMISSION_TO_CREATE_TABLE, severity=Severity.HIGH, version=1)


class CreateSchemaNotPermittedAlert(DatasetAlert):
    """
    Concrete alert class when user does not have permission to create schemas.
    """

    def __init__(self):
        super().__init__(
            name=Tag.ALERT_NO_PERMISSION_TO_CREATE_SCHEMA, severity=Severity.HIGH, version=1)


class NoFeatureColumnsAlert(DatasetAlert):
    """
    Concrete class for no feature columns. This happens when AutoML is run with a dataset that does not have any feature
    columns with supported types.
    """

    def __init__(self):
        super().__init__(name=Tag.ALERT_NO_FEATURE_COLUMNS, severity=Severity.HIGH, version=1)


class DatasetTooLargeAlert(DatasetAlert):
    """
    Emitted when the dataset is too large for single node training, and AutoML samples the dataset.
    """

    def __init__(self):
        super().__init__(name=Tag.ALERT_DATASET_TOO_LARGE, severity=Severity.MEDIUM, version=2)


class DatasetTruncatedAlert(DatasetAlert):
    """
    Emitted when AutoML truncates the dataset for forecasting.
    """

    def __init__(self):
        super().__init__(name=Tag.ALERT_DATASET_TRUNCATED, severity=Severity.MEDIUM, version=1)


class DatasetEmptyAlert(DatasetAlert):
    """
    Emitted when the dataset does not have any rows.
    """

    def __init__(self):
        super().__init__(name=Tag.ALERT_DATASET_EMPTY, severity=Severity.HIGH, version=1)


class NotEnoughHistoricalDataAlert(DatasetAlert):
    """
    Emitted when there is not enough data in the timeseries. The user should provide more data.
    """

    def __init__(self):
        super().__init__(
            name=Tag.ALERT_NOT_ENOUGH_HISTORICAL_DATA, severity=Severity.HIGH, version=2)


class EarlyStopAlert(DatasetAlert):
    """
    Concrete class for early stopping. This happens when hyperopt run in AutoML reaches the early stopping condition
    and stopped.
    """

    def __init__(self):
        super().__init__(name=Tag.ALERT_TRAINING_EARLY_STOPPED, severity=Severity.LOW, version=1)


class DataExplorationFailAlert(DatasetAlert):
    """
    Emitted when the data exploration notebook fails.
    """

    def __init__(self):
        super().__init__(name=Tag.ALERT_DATA_EXPLORATION_FAIL, severity=Severity.MEDIUM, version=1)


class DataExplorationTruncateRowsAlert(DatasetAlert):
    """
    Emitted when data exploration notebook truncate rows in the dataset.
    """

    def __init__(self):
        super().__init__(
            name=Tag.ALERT_DATA_EXPLORATION_TRUNCATE_ROWS, severity=Severity.LOW, version=1)


class DataExplorationTruncateColumnsAlert(DatasetAlert):
    """
    Emitted when data exploration notebook truncate columns in the dataset.
    """

    def __init__(self):
        super().__init__(
            name=Tag.ALERT_DATA_EXPLORATION_TRUNCATE_COLUMNS, severity=Severity.LOW, version=1)


class UnableToSampleWithoutSkewAlert(DatasetAlert):
    """
    Emitted when we are unable to sample the dataset without skewing the target label distribution
    """

    def __init__(self):
        super().__init__(
            name=Tag.ALERT_UNABLE_TO_SAMPLE_WITHOUT_SKEW, severity=Severity.HIGH, version=1)


class ExecutionTimeoutAlert(DatasetAlert):
    """
    Emitted when AutoML times out before any trials are completed
    """

    def __init__(self):
        super().__init__(name=Tag.ALERT_EXECUTION_TIMEOUT, severity=Severity.HIGH, version=1)


class TargetLabelImbalanceAlert(DatasetAlert):
    """
    Emitted when the counts of the least and most frequent target classes have a ratio <10%
    """

    def __init__(self, most_frequent_label: ClassificationTargetTypes,
                 least_frequent_label: ClassificationTargetTypes, ratio: float):
        super().__init__(
            name=Tag.ALERT_TARGET_LABEL_IMBALANCE,
            severity=Severity.MEDIUM,
            version=1,
            additional_info={
                "most_frequent_label": str(most_frequent_label),
                "least_frequent_label": str(least_frequent_label),
                "ratio": f"{ratio:.3f}"
            })


class TargetLabelRatioAlert(DatasetAlert):
    """
    Emitted when the counts of the least and most frequent target classes have a ratio >=10%
    """

    def __init__(self, most_frequent_label: ClassificationTargetTypes,
                 least_frequent_label: ClassificationTargetTypes, ratio: float):
        super().__init__(
            name=Tag.ALERT_TARGET_LABEL_RATIO,
            severity=Severity.LOW,
            version=1,
            additional_info={
                "most_frequent_label": most_frequent_label,
                "least_frequent_label": least_frequent_label,
                "ratio": f"{ratio:.3f}"
            })


class InappropriateMetricForImbalanceAlert(DatasetAlert):
    """
    Emitted when a classification metric that's not suitable for an imbalanced dataset was chosen (e.g., accuracy)
    """

    def __init__(self, most_frequent_label: ClassificationTargetTypes,
                 least_frequent_label: ClassificationTargetTypes, ratio: float, metric: str,
                 appropriate_metric: str):
        super().__init__(
            name=Tag.ALERT_INAPPROPRIATE_METRIC_FOR_IMBALANCE,
            severity=Severity.HIGH,
            version=1,
            additional_info={
                "most_frequent_label": most_frequent_label,
                "least_frequent_label": least_frequent_label,
                "ratio": f"{ratio:.3f}",
                "metric": metric,
                "appropriate_metric": appropriate_metric
            })
