from typing import Any, Dict, List, Set, Union, Optional

from databricks.automl.legacy.alerts.alert import Alert, Severity, AlertDisplay, AlertDisplayGroup
from databricks.automl.legacy.const import SemanticType
from databricks.automl.legacy.problem_type import ClassificationTargetTypes
from databricks.automl.legacy.tags import Tag


class FeatureAlert(Alert):
    def __init__(self,
                 name: Tag,
                 affected_ids: Union[Set[str], List[str], Dict[str, str]],
                 severity: Severity,
                 version: int,
                 additional_info: Dict[str, Any] = None):
        super().__init__(
            name=name, severity=severity, version=version, additional_info=additional_info)
        if isinstance(affected_ids, list) or isinstance(affected_ids, set):
            ids = [{Alert.COL_NAME: id, Alert.COL_TYPE: None} for id in sorted(affected_ids)]
        else:
            ids = [{
                Alert.COL_NAME: id,
                Alert.COL_TYPE: id_type
            } for id, id_type in affected_ids.items()]

        self._misc = {
            Alert.AFFECTED_IDS: ids,
        }

    @property
    def misc(self) -> Dict[str, Any]:
        return self._misc


class TimeSeriesIdentitiesTooShortAlert(FeatureAlert):
    """
    Emitted when there exist identities for forecasting that are too short. The user must provide more data.
    """

    def __init__(self, identities: List[str]):
        super().__init__(
            name=Tag.ALERT_TIME_SERIES_IDENTITIES_TOO_SHORT,
            affected_ids=identities,
            severity=Severity.HIGH,
            version=2)


class TruncateHorizonAlert(FeatureAlert):
    """
    Emitted when the user-provided horizon is too long relative to some timeseries.
    """

    def __init__(self, identities: List[str]):
        super().__init__(
            name=Tag.ALERT_TRUNCATE_HORIZON,
            affected_ids=identities,
            severity=Severity.LOW,
            version=1)


class UnsupportedTargetTypeAlert(FeatureAlert):
    """
    Concrete class for unsupported target type. This alert will be emitted when a target column has an unsupported type.
    """

    def __init__(self, col_name: str, col_type: str):
        super().__init__(
            name=Tag.ALERT_UNSUPPORTED_TARGET_TYPE,
            affected_ids={col_name: col_type},
            severity=Severity.HIGH,
            version=1)


class UnsupportedTimeTypeAlert(FeatureAlert):
    """
    Concrete class for unsupported time type. This alert will be emitted when the time column provided to forecasting is
    invalid
    """

    def __init__(self, col_name: str, col_type: str):
        super().__init__(
            name=Tag.ALERT_UNSUPPORTED_TIME_TYPE,
            affected_ids={col_name: col_type},
            severity=Severity.HIGH,
            version=1)


class ConstantColumnAlert(FeatureAlert):
    """
    Concrete alert class for constant columns from pandas-profiling.
    As of version 2, AutoML drops these columns.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_CONSTANT_COLUMNS,
            affected_ids=col_names,
            severity=Severity.LOW,
            version=2,
        )


class ExtremeCardinalityColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns with high cardinality from pandas-profiling.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_EXTREME_CARDINALITY_COLUMNS,
            affected_ids=col_names,
            severity=Severity.MEDIUM,
            version=1)


class HighCardinalityColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns with high cardinality from pandas-profiling.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_HIGH_CARDINALITY_COLUMNS,
            affected_ids=col_names,
            severity=Severity.LOW,
            version=1)


class SingleClassInTargetColumnAlert(FeatureAlert):
    """
    Concrete alert class for target column that contains only 1 distinct target class.
    """

    def __init__(self, col_name: str):
        super().__init__(
            name=Tag.ALERT_LOW_CARDINALITY_TARGET_COLUMN,
            affected_ids=[col_name],
            severity=Severity.HIGH,
            version=1)


class NullsInTargetColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns with nulls in target column.
    """

    def __init__(self, col_name: str):
        super().__init__(
            name=Tag.ALERT_NULLS_IN_TARGET_COLUMN,
            affected_ids=[col_name],
            severity=Severity.MEDIUM,
            version=2)


class NullsInTimeColumnAlert(FeatureAlert):
    """
    Emitted when the time column provided to supervised_learner has nulls.
    """

    def __init__(self, col_name: str, col_type: str):
        super().__init__(
            name=Tag.ALERT_NULLS_IN_TIME_COLUMN,
            affected_ids={col_name: col_type},
            severity=Severity.MEDIUM,
            version=1)


class SmallNullsColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns with nulls less than a threshold.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_SMALL_NULLS_COLUMNS,
            affected_ids=col_names,
            severity=Severity.LOW,
            version=1)


class LargeNullsColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns with nulls more than a threshold.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_LARGE_NULLS_COLUMNS,
            affected_ids=col_names,
            severity=Severity.MEDIUM,
            version=1)


class HighCorrelationColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns have high correlations with other columns.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_HIGH_CORRELATION_COLUMNS,
            affected_ids=col_names,
            severity=Severity.LOW,
            version=1)


class SkewedColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns that are highly skewed.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_SKEWED_COLUMNS, affected_ids=col_names, severity=Severity.LOW, version=1)


class UniformColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns that are uniformly distributed.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_UNIFORM_COLUMNS,
            affected_ids=col_names,
            severity=Severity.LOW,
            version=1)


class UniqueColumnAlert(FeatureAlert):
    """
    Concrete alert class for columns that contains unique values
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_UNIQUE_COLUMNS, affected_ids=col_names, severity=Severity.LOW, version=1)


class UniqueStringColumnAlert(FeatureAlert):
    """
    Concrete alert class for string columns whose values are all unique.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_UNIQUE_STRING_COLUMNS,
            affected_ids=col_names,
            severity=Severity.LOW,
            version=1)


class UnsupportedColumnAlert(FeatureAlert):
    """
    Concrete alert class for unsupported columns
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_UNSUPPORTED_FEATURE_COLS,
            affected_ids=col_names,
            severity=Severity.HIGH,
            version=1)
        self._col_names = col_names

    def display(self) -> Optional[List[AlertDisplay]]:
        cols = ", ".join([f"`{col}`" for col in self._col_names])
        return [
            AlertDisplay(
                string=f"Columns {cols} are not supported and will be dropped before running " +
                "the data exploration and training notebooks.",
                group=AlertDisplayGroup.OTHER)
        ]


class IncompatibleSemanticTypeAnnotationAlert(FeatureAlert):
    """
    Concrete alert class for incompatible semantic type annotations
    """

    def __init__(self, annotations: Dict[str, List[str]]):
        super().__init__(
            name=Tag.ALERT_INCOMPATIBLE_ANNOTATION,
            # we only need the keys (column names) to display on the alerts dashboard
            affected_ids=list(annotations.keys()),
            severity=Severity.MEDIUM,
            version=1)

        self._annotations = annotations

    def display(self) -> Optional[List[AlertDisplay]]:
        alert_displays = []
        for col, tpe in self._annotations.items():
            tpe = ", ".join([f"`{t}`" for t in tpe])
            alert_display = AlertDisplay(
                string=f"Column `{col}` was annotated as semantic type {tpe}, " +
                "but it's type is incompatible, and the annotation will be ignored.",
                group=AlertDisplayGroup.SEMANTIC_TYPE)
            alert_displays.append(alert_display)
        return alert_displays


# For type detection alerts, keep the text consistent with
# mlflow/web/js/src/experiment-tracking/components/automl/AutoMLAlertText.js
def create_semantic_type_message(col_names: List[str], semantic_type: SemanticType) -> str:
    cols = ", ".join([f"`{col}`" for col in col_names])
    plural = "s" if len(col_names) > 1 else ""
    base_message = f"Semantic type `{semantic_type.value}` detected for column{plural} {cols}."
    if semantic_type == SemanticType.TEXT:
        return f"{base_message} Training notebooks will convert each column into a fixed-length feature vector."
    elif semantic_type == SemanticType.DATETIME:
        return f"{base_message} Training notebooks will convert each column to a datetime type and encode features based on temporal transformations."
    elif semantic_type == SemanticType.NUMERIC:
        return f"{base_message} Training notebooks will convert each column to a numeric type and encode features based on numerical transformations."
    elif semantic_type == SemanticType.CATEGORICAL:
        return f"{base_message} Training notebooks will encode features based on categorical transformations."
    else:
        return f"{base_message} Columns will be automatically converted in the data exploration and training notebooks."


class StrongDatetimeSemanticTypeDetectionAlert(FeatureAlert):
    """
    Concrete alert class for strong detection of datetime semantic type
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_STRONG_DATETIME_TYPE_DETECTION,
            affected_ids=col_names,
            severity=Severity.MEDIUM,
            version=1)
        self._col_names = col_names

    def display(self) -> Optional[List[AlertDisplay]]:
        return [
            AlertDisplay(
                string=create_semantic_type_message(self._col_names, SemanticType.DATETIME),
                group=AlertDisplayGroup.SEMANTIC_TYPE)
        ]


class StrongNumericSemanticTypeDetectionAlert(FeatureAlert):
    """
    Concrete alert class for strong detection of numeric semantic type
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_STRONG_NUMERIC_TYPE_DETECTION,
            affected_ids=col_names,
            severity=Severity.MEDIUM,
            version=1)
        self._col_names = col_names

    def display(self) -> Optional[List[AlertDisplay]]:
        return [
            AlertDisplay(
                string=create_semantic_type_message(self._col_names, SemanticType.NUMERIC),
                group=AlertDisplayGroup.SEMANTIC_TYPE)
        ]


class StrongCategoricalSemanticTypeDetectionAlert(FeatureAlert):
    """
    Concrete alert class for strong detection of categorical semantic type
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_STRONG_CATEGORICAL_TYPE_DETECTION,
            affected_ids=col_names,
            severity=Severity.MEDIUM,
            version=1)
        self._col_names = col_names

    def display(self) -> Optional[List[AlertDisplay]]:
        return [
            AlertDisplay(
                string=create_semantic_type_message(self._col_names, SemanticType.CATEGORICAL),
                group=AlertDisplayGroup.SEMANTIC_TYPE)
        ]


class StrongTextSemanticTypeDetectionAlert(FeatureAlert):
    """
    Concrete alert class for strong detection of text semantic type
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_STRONG_TEXT_TYPE_DETECTION,
            affected_ids=col_names,
            severity=Severity.MEDIUM,
            version=1)
        self._col_names = col_names

    def display(self) -> Optional[List[AlertDisplay]]:
        return [
            AlertDisplay(
                string=create_semantic_type_message(self._col_names, SemanticType.TEXT),
                group=AlertDisplayGroup.SEMANTIC_TYPE)
        ]


class AllRowsInvalidAlert(FeatureAlert):
    """
    Emitted when every value in the target column is null. AutoML will raise an UnsupportedDataError.
    """

    def __init__(self, col_name: str):
        super().__init__(
            name=Tag.ALERT_ALL_ROWS_INVALID,
            affected_ids=[col_name],
            severity=Severity.HIGH,
            version=1)


class DuplicateColumnNamesAlert(FeatureAlert):
    """
    Emitted when there are columns in the dataset with the same name. AutoML will raise an UnsupportedDataError.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_DUPLICATE_COLUMN_NAMES,
            affected_ids=col_names,
            severity=Severity.HIGH,
            version=1)


class ArrayNotNumericalAlert(FeatureAlert):
    """
    Emitted when the arrays are not of numerical type. AutoML will drop them in training.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_ARRAY_NOT_NUMERICAL,
            affected_ids=col_names,
            severity=Severity.LOW,
            version=1)


class ArrayNotSameLengthAlert(FeatureAlert):
    """
    Emitted when the arrays are not of the same length. AutoML will drop them in training.
    """

    def __init__(self, col_names: List[str]):
        super().__init__(
            name=Tag.ALERT_ARRAY_NOT_SAME_LENGTH,
            affected_ids=col_names,
            severity=Severity.LOW,
            version=1)


class TargetLabelInsufficientDataAlert(FeatureAlert):
    """
    Emitted with the list of class names for target labels that don't have enough data to run AutoML
    """

    def __init__(self, class_names: List[ClassificationTargetTypes]):
        super().__init__(
            name=Tag.ALERT_TARGET_LABEL_INSUFFICIENT_DATA,
            affected_ids=[str(c) for c in class_names],
            severity=Severity.MEDIUM,
            version=1)

    def __str__(self):
        return self._misc


class MissingTimeStepsInTimeSeriesAlert(FeatureAlert):
    """
    Emitted when the time_col in forecasting has missing time steps compared to given frequency.
    """

    def __init__(self, time_col: str):
        super().__init__(
            name=Tag.ALERT_MISSING_TIME_STEPS_IN_TIME_SERIES,
            affected_ids=[time_col],
            severity=Severity.LOW,
            version=1)


class ExtraTimeStepsInTimeSeriesAlert(FeatureAlert):
    """
    Emitted when the time_col in forecasting has extra time steps compared to given frequency.
    """

    def __init__(self, time_col: str):
        super().__init__(
            name=Tag.ALERT_EXTRA_TIME_STEPS_IN_TIME_SERIES,
            affected_ids=[time_col],
            severity=Severity.MEDIUM,
            version=1)


class UnmatchedFrequencyInTimeSeriesAlert(FeatureAlert):
    """
    Emitted when the detected frequency in time_col does not match specified frequency in forecasting. This alert will
    not be recorded as the same time with MissingTimeStepsInTimeSeriesAlert or ExtraTimeStepsInTimeSeriesAlert. In
    other words, the alert also indicates that the time series itself is uniformly spaced.
    """

    def __init__(self, time_col: str):
        super().__init__(
            name=Tag.ALERT_UNMATCHED_FREQUENCY_IN_TIME_SERIES,
            affected_ids=[time_col],
            severity=Severity.MEDIUM,
            version=1)


class InferredPosLabelAlert(FeatureAlert):
    """
    Emitted when classify() is called for binary classification without specifying pos_label.
    """

    def __init__(self, pos_label: str):
        super().__init__(
            name=Tag.ALERT_INFERRED_POS_LABEL,
            affected_ids=[pos_label],
            severity=Severity.LOW,
            version=1)
