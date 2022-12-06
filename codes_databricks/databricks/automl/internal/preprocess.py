import logging
from dataclasses import dataclass
from typing import Type, Mapping

from pyspark.sql.types import DataType, StringType, DateType, StructType

from databricks.automl.internal.alerts import NoFeatureColumnsAlert
from databricks.automl.internal.alerts.alert_handler import FeatureAlertsHandler
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.alerts.feature_alert import *
from databricks.automl.internal.base_preprocess import BaseDataPreprocessor
from databricks.automl.internal.common.const import AutoMLDataType, \
    SparseOrDense, SparkDataType
from databricks.automl.internal.errors import UnsupportedColumnError
from databricks.automl.internal.sections.training.preprocess import CategoricalPreprocessor
from databricks.automl.internal.stats import IntermediateStats, PostSamplingStats
from databricks.automl.shared.const import SemanticType, ProblemType
from databricks.automl.shared.errors import UnsupportedDataError

_logger = logging.getLogger(__name__)


# NOTE: this data class should ideally be put in size_estimator.py, but it caused a circular dependency
# to the `SupervisedLearnerDataPreprocessConfs` class below.
# TODO(ML-21514): consider moving SupervisedLearnerDataPreprocessConfs to conf.py
@dataclass
class SizeEstimatorResult:
    # Whether the sparse matrix decide to encode the data as sparse or dense matrix.
    sparse_or_dense: SparseOrDense
    # Memory required to load the full dataset in MB.
    mem_req_data_load_mb: float
    # Memory required to train the full dataset with dense encoding.
    mem_req_training_mb_dense: float
    # Memory required to train the full dataset with sparse encoding.
    mem_req_training_mb_sparse: float
    # Sample fraction estimated in the precise sampling step for the training to be successful.
    precise_sample_fraction: Optional[float]
    # Final sample fraction that combines both rough and precise sampling steps.
    final_sample_fraction: Optional[float]


@dataclass
class SupervisedLearnerDataPreprocessResults:
    """
    Class for the data preprocess results
    """
    multiclass: bool
    categorical_numerical_columns_low_cardinality: Set[str]
    numerical_columns: Set[str]  # Columns interpreted as numerics -- disjoint from previous two
    string_columns_low_cardinality: Set[str]
    string_columns_high_cardinality: Set[str]
    string_columns_extreme_cardinality: Set[str]
    string_columns_unique_values: Set[str]
    text_columns: Set[str]
    array_columns: Set[str]
    boolean_columns: Set[str]
    unsupported_columns: Set[str]
    constant_columns: Set[str]
    date_columns: Set[str]
    timestamp_columns: Set[str]
    num_nulls: Dict[str, int]
    target_col_type: Type[DataType]
    strong_semantic_detections: Dict[SemanticType, List[str]]
    size_estimator_result: SizeEstimatorResult
    num_classes: Optional[int]  # only populated for classification
    has_feature_store_joins: bool = False


@dataclass
class SupervisedLearnerDataPreprocessConfs:
    """
    Configurations for the data preprocess
    """
    DEFAULT_CATEGORICAL_HIGH_CARDINALITY_THRESHOLD = CategoricalPreprocessor.DEFAULT_HASH_OUTPUT_COLS
    DEFAULT_CATEGORICAL_EXTREME_CARDINALITY_THRESHOLD = 65535

    categorical_high_cardinality_threshold: int
    categorical_extreme_cardinality_threshold: int

    def __init__(
            self,
            categorical_high_cardinality_threshold=DEFAULT_CATEGORICAL_HIGH_CARDINALITY_THRESHOLD,
            categorical_extreme_cardinality_threshold=DEFAULT_CATEGORICAL_EXTREME_CARDINALITY_THRESHOLD
    ):
        self.categorical_high_cardinality_threshold = categorical_high_cardinality_threshold
        self.categorical_extreme_cardinality_threshold = categorical_extreme_cardinality_threshold


class SupervisedLearnerDataPreprocessor(BaseDataPreprocessor):
    """
    Validate the dataset and extract information from it
    """

    CONFS = SupervisedLearnerDataPreprocessConfs()
    SUPPORTED_ARRAY_ELEMENT_TYPES = [AutoMLDataType.NUMERIC]

    def __init__(self,
                 intermediate_stats: IntermediateStats,
                 dataset_schema: StructType,
                 target_col: str,
                 supported_target_types: List[DataType],
                 alert_manager: AlertManager,
                 time_col: Optional[str] = None,
                 supported_time_types: List[DataType] = SparkDataType.TIME_TYPES,
                 confs: SupervisedLearnerDataPreprocessConfs = CONFS):
        super().__init__(
            intermediate_stats=intermediate_stats,
            dataset_schema=dataset_schema,
            alert_manager=alert_manager,
            target_col=target_col,
        )

        # initialize helper module to generate alerts
        self._feature_alert_handler = FeatureAlertsHandler(alert_manager)

        self._categorical_high_cardinality_threshold = confs.categorical_high_cardinality_threshold
        self._categorical_extreme_cardinality_threshold = confs.categorical_extreme_cardinality_threshold

        self._supported_cols = intermediate_stats.supported_cols
        self._feature_schema = intermediate_stats.feature_schema
        self._target_col_type = self._validate_col_type(target_col, "target_col",
                                                        supported_target_types)
        self._target_col = target_col
        self._time_col_type = self._validate_col_type(time_col, "time_col",
                                                      supported_time_types) if time_col else None

    @property
    def selected_cols(self) -> List[str]:
        return self._supported_cols

    @property
    def target_col_type(self) -> Type[DataType]:
        return self._target_col_type

    @property
    def time_col_type(self) -> Optional[Type[DataType]]:
        return self._time_col_type

    @property
    def feature_schema(self) -> Mapping[Union[str, Type[DataType]], List[str]]:
        return self._feature_schema

    def _validate_array_columns(self, array_columns: Set[str], stats: PostSamplingStats):
        """
        Validate whether the array columns are supported or not.

        This function splits the input `array_columns` into "supported" and "unsupported" sets,
        and log warnings to warning dashboard for unsupported reasons.

        :param array_columns: a set of array columns to be validated
        :param stats: the post sampling stats from the previous step
        :return: tuple(supported_array_columns, unsupported_array_columns)
        """
        supported_array_columns = set()
        not_numerical_array_columns = set()
        not_same_length_array_columns = set()
        for column in array_columns:
            array_stats = stats.columns[column]
            if array_stats.elementType not in self.SUPPORTED_ARRAY_ELEMENT_TYPES:
                not_numerical_array_columns.add(column)
                continue
            if array_stats.min_length != array_stats.max_length:
                not_same_length_array_columns.add(column)
                continue
            supported_array_columns.add(column)

        warnings = []
        if not_numerical_array_columns:
            warnings.append(ArrayNotNumericalAlert(not_numerical_array_columns))
        if not_same_length_array_columns:
            warnings.append(ArrayNotSameLengthAlert(not_same_length_array_columns))
        if warnings:
            self._feature_alert_handler.log_warnings(warnings)

        unsupported_array_columns = not_numerical_array_columns | not_same_length_array_columns
        return supported_array_columns, unsupported_array_columns

    def log_feature_alerts(self, data_exp_result: Dict[str, Any]) -> None:
        """
        Log feature alerts generated by the data exploration notebook.
        """
        if "alerts" in data_exp_result.keys():
            warnings = self._feature_alert_handler.aggregate_warnings(data_exp_result["alerts"])
            self._feature_alert_handler.log_warnings(warnings)

    def process_post_sampling_stats(
            self, stats: PostSamplingStats, target_col: str, problem_type: ProblemType,
            size_estimator_result: SizeEstimatorResult) -> SupervisedLearnerDataPreprocessResults:
        multiclass = None
        num_classes = None
        if problem_type == ProblemType.CLASSIFICATION:
            num_classes = stats.columns[target_col].num_distinct
            multiclass = num_classes > 2
            if num_classes < 2:
                self._alert_manager.record(SingleClassInTargetColumnAlert(target_col))
                raise UnsupportedColumnError(
                    "Target column must contain at least 2 distinct target classes")

        # column_types: map from types to columns with that type, excluding target column
        # eg. {AutoMLDataType.NUMERIC: set(["col_1", "col_2"])}
        column_types = {}
        for name, column in stats.columns.items():
            if name != target_col:
                column_types.setdefault(column.type, set()).add(name)

        constant_columns = set([name for name, col in stats.columns.items() if col.is_constant])

        numerical_cols = column_types.get(AutoMLDataType.NUMERIC, set())
        categorical_numerical_columns_low_cardinality = set()  # these will be one-hot encoded
        string_columns_low_cardinality = set()
        string_columns_high_cardinality = set()
        string_columns_extreme_cardinality = set()
        string_columns_unique_values = set()

        for col in column_types.get(AutoMLDataType.STRING, []):
            if stats.columns[col].num_distinct:
                n_distinct = stats.columns[col].num_distinct
            else:
                n_distinct = stats.columns[col].approx_num_distinct

            if col in self.feature_schema[SparkDataType.NUMERICAL_TYPE]:
                numerical_cols.add(col)
                if n_distinct < self._categorical_high_cardinality_threshold \
                        and col not in constant_columns:
                    categorical_numerical_columns_low_cardinality.add(col)
            elif col in self.feature_schema[StringType]:
                # Note: if each value of a string column is unique, we mark it as 'UNIQUE_STRINGS',
                # and then it will not have any of the '[EXTREME,HIGH,LOW]_CARDINALITY' tag.
                if n_distinct == stats.num_rows:
                    string_columns_unique_values.add(col)
                elif n_distinct >= self._categorical_extreme_cardinality_threshold:
                    string_columns_extreme_cardinality.add(col)
                elif n_distinct >= self._categorical_high_cardinality_threshold:
                    string_columns_high_cardinality.add(col)
                elif col not in constant_columns:
                    string_columns_low_cardinality.add(col)

        text_columns = column_types.get(AutoMLDataType.TEXT, set()) - constant_columns
        numerical_cols = numerical_cols - constant_columns

        boolean_columns = column_types.get(AutoMLDataType.BOOLEAN, set()) - constant_columns
        supported_array_columns, unsupported_array_columns = self._validate_array_columns(
            column_types.get(AutoMLDataType.ARRAY, set()), stats)

        # Calculate Timestamp columns
        # feature_schema[TimestampType] will not include columns we detected/converted before
        # running pandas-profiling, and we need to subtract DateType columns so they're not
        # double counted (pandas-profiling DateTime will include DateType and TimestampType)
        timestamp_columns = column_types.get(AutoMLDataType.DATETIME, set()) - set(
            self.feature_schema[DateType]) - constant_columns

        num_nulls = {name: col.num_missing for name, col in stats.columns.items()}

        if string_columns_extreme_cardinality:
            _logger.warning("The following string columns with too many distinct values will be " +
                            f"dropped by AutoML: {string_columns_extreme_cardinality}. ")
        if string_columns_unique_values:
            _logger.warning("The following string columns with unique values will be " +
                            f"dropped by AutoML: {string_columns_unique_values}. ")

        if len(string_columns_unique_values | string_columns_extreme_cardinality |
               set(self.feature_schema[SparkDataType.UNSUPPORTED_TYPE])) >= len(
                   self._dataset_schema.fields) - 1:
            self._alert_manager.record(NoFeatureColumnsAlert())
            raise UnsupportedDataError(
                f"No supported columns found in dataset. Columns types are not supported. "
                f"Or string columns contains too many distinct values. "
                f"Please use string columns with less than "
                f"{self._categorical_extreme_cardinality_threshold} distinct values.")

        warnings = []
        if string_columns_extreme_cardinality:
            warnings.append(ExtremeCardinalityColumnAlert(string_columns_extreme_cardinality))
        if string_columns_high_cardinality:
            warnings.append(HighCardinalityColumnAlert(string_columns_high_cardinality))
        if string_columns_unique_values:
            warnings.append(UniqueStringColumnAlert(string_columns_unique_values))
        if constant_columns:
            warnings.append(ConstantColumnAlert(constant_columns))

        self._feature_alert_handler.log_warnings(warnings)

        preprocess_result = SupervisedLearnerDataPreprocessResults(
            multiclass=multiclass,
            categorical_numerical_columns_low_cardinality=
            categorical_numerical_columns_low_cardinality,
            numerical_columns=numerical_cols,
            string_columns_low_cardinality=string_columns_low_cardinality,
            string_columns_high_cardinality=string_columns_high_cardinality,
            string_columns_extreme_cardinality=string_columns_extreme_cardinality,
            string_columns_unique_values=string_columns_unique_values,
            text_columns=text_columns,
            array_columns=supported_array_columns,
            boolean_columns=boolean_columns,
            unsupported_columns=set(self.feature_schema[SparkDataType.UNSUPPORTED_TYPE]) |
            unsupported_array_columns,
            constant_columns=constant_columns,
            date_columns=set(self.feature_schema[DateType]),
            timestamp_columns=timestamp_columns,
            num_nulls=num_nulls,
            target_col_type=self._target_col_type,
            strong_semantic_detections=None,
            size_estimator_result=size_estimator_result,
            num_classes=num_classes)

        return preprocess_result
