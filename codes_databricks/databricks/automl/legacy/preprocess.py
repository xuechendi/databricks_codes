import importlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple, Iterable, Mapping, Set

import fasttext
import numpy as np
import pandas as pd
import pandas.api.types as pd_types
from pyspark.databricks.sql import annotation_utils
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, StringType, DateType, StructType, TimestampType

from databricks.automl.legacy import artifacts
from databricks.automl.legacy.alerts import NoFeatureColumnsAlert, SingleClassInTargetColumnAlert, \
    IncompatibleSemanticTypeAnnotationAlert
from databricks.automl.legacy.alerts.alert_handler import FeatureAlertsHandler
from databricks.automl.legacy.alerts.alert_manager import AlertManager
from databricks.automl.legacy.alerts.feature_alert import *
from databricks.automl.legacy.base_preprocess import BaseDataPreprocessor
from databricks.automl.legacy.const import AutoMLDataType, DatasetFormat, SemanticType, \
    SparseOrDense
from databricks.automl.legacy.errors import UnsupportedDataError, UnsupportedColumnError
from databricks.automl.legacy.io_utils import filter_stderr
from databricks.automl.legacy.problem_type import ProblemType
from databricks.automl.legacy.sections.training.preprocess import CategoricalPreprocessor
from databricks.automl.legacy.stats import PreSamplingStats, PostSamplingStats

_logger = logging.getLogger(__name__)


# NOTE: this data class should ideally be put in size_estimator.py, but it caused a circular dependency
# to the `SupervisedLearnerDataPreprocessConfs` class below.
# TODO(ML-21514): consider moving SupervisedLearnerDataPreprocessConfs to conf.py
@dataclass
class SizeEstimatorResult:
    # Whether the sparse matrix decide to encode the data as sparse or dense matrix.
    sparse_or_dense: SparseOrDense
    # Memory required to load the full datset in MB.
    mem_req_data_load_mb: float
    # Memory required to train the full dataset with dense encoding.
    mem_req_training_mb_dense: float
    # Memory required to train the full dataset with sparse encoding.
    mem_req_training_mb_sparse: float
    # Estimated sample fraction for the training to be successful.
    sample_fraction: Optional[float]


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
    """
    Threshold values for numeric vs. categorical empirically determined by using benchmark columns:
    https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#notebook/541837113415607
    STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD bumped from 10 -> 12 
    in order to accommodate months encoded as integers.
    
    Threshold values for text detection empirically determined using English text column benchmarks:
    https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#notebook/4050779996430146
    """
    # Boundary between numeric and categorical detection
    NUMERIC_CATEGORICAL_CARDINALITY_THRESHOLD = 40
    # Boundary between a strong and weak categorical detection
    STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD = 12
    # Boundary between a strong and weak numeric detection
    STRONG_NUMERIC_DETECTION_CARDINALITY_THRESHOLD = 250
    # Sample size to use when running semantic type detections
    TYPE_DETECTION_SAMPLE_SIZE = 10_000
    # Name of fasttext language identification model file in databricks.automl.artifacts package
    FASTTEXT_MODEL_FILE = "lid.176.ftz"
    # Label for English detections in fasttext
    FASTTEXT_ENGLISH_LABEL = "__label__en"
    # Confidence to determine something to be English
    FASTTEXT_CONFIDENCE_THRESHOLD = 0.75
    # Percentage of distinct values in a column required to consider a column for text detection
    TEXT_PERCENT_DISTINCT_THRESHOLD = 0.5

    def __init__(self,
                 pre_sampling_stats: PreSamplingStats,
                 dataset_schema: StructType,
                 target_col: str,
                 supported_target_types: List[DataType],
                 alert_manager: AlertManager,
                 time_col: Optional[str] = None,
                 supported_time_types: List[DataType] = BaseDataPreprocessor.TIME_TYPES,
                 confs: SupervisedLearnerDataPreprocessConfs = CONFS,
                 dataset_format: DatasetFormat = DatasetFormat.SPARK):
        super().__init__(
            pre_sampling_stats=pre_sampling_stats,
            dataset_schema=dataset_schema,
            dataset_format=dataset_format,
            alert_manager=alert_manager,
            target_col=target_col,
        )

        # initialize helper module to generate alerts
        self._feature_alert_handler = FeatureAlertsHandler(alert_manager)

        self._categorical_high_cardinality_threshold = confs.categorical_high_cardinality_threshold
        self._categorical_extreme_cardinality_threshold = confs.categorical_extreme_cardinality_threshold

        self._schema_info = self._get_schema_info()
        self._target_col_type = self._validate_col_type(target_col, "target_col",
                                                        supported_target_types)
        self._target_col = target_col
        self._time_col_type = self._validate_col_type(time_col, "time_col",
                                                      supported_time_types) if time_col else None

        self._strong_semantic_detections = defaultdict(list)  # Maps semantic types to column names
        self._weak_semantic_detections = defaultdict(list)  # Maps semantic types to column names

        _logger.debug(f"Schema Info: {self._schema_info}")

    @property
    def selected_cols(self) -> List[str]:
        return self._schema_info.supported_cols

    @property
    def target_col_type(self) -> Type[DataType]:
        return self._target_col_type

    @property
    def time_col_type(self) -> Optional[Type[DataType]]:
        return self._time_col_type

    @property
    def feature_schema(self) -> Mapping[Union[str, Type[DataType]], List[str]]:
        return self._schema_info.feature_schema

    @property
    def strong_semantic_detections(self) -> Dict[SemanticType, List[str]]:
        """Dictionary mapping semantic type to list of column names with strong detection."""
        return self._strong_semantic_detections

    @property
    def weak_semantic_detections(self) -> Dict[SemanticType, List[str]]:
        """Dictionary mapping semantic type to list of column names with weak detection."""
        return self._weak_semantic_detections

    @staticmethod
    def _detect_semantic_type_by_conversion(df: pd.DataFrame,
                                            convert_fn: Callable[[pd.Series], pd.Series],
                                            **convert_kwargs: Any) -> List[str]:
        """
        Returns all columns in a pd.DataFrame that are strongly detected to have semantic type
        defined by the conversion function. These columns must have at least one successful
        conversion, and no more than one distinct non-null, non-convertable value.

        :param df: pandas DataFrame of columns to attempt conversion
        :param convert_fn: data type conversion function to apply to each column, must coerce
                           conversion errors to null
        :param convert_kwargs: extra keyword args to pass to convert_fn
        :return: list of columns that meet the detection criteria
        """
        detections = []
        for column in df.columns:
            value_set = pd.Series(data=df[column].dropna().unique(), name=column)
            try:
                result = convert_fn(value_set, **convert_kwargs)
            except Exception as e:
                # NOTE: `convert_fn` might fail due to pandas internal bug.
                # See ES-380781 for more details.
                _logger.error("Conversion function failed: ", e)
                continue

            # No successful conversions or multiple non-convertable (i.e. missing) values
            errors = result.isna()
            if errors.all() or errors.sum() > 1:
                continue

            detections.append(column)
        return detections

    def _filter_cols_with_semantic_types(self, cols: Iterable[str]) -> List[str]:
        """
        Filters the input columns and removes those that are associated with a semantic type.

        :param cols: names of input cols
        :return: names of cols that do not currently carry a semantic type
        """
        cols_with_semantic_types = list(chain(*self.strong_semantic_detections.values()))
        return [col for col in cols if col not in cols_with_semantic_types]

    def _add_strong_detections(self, semantic_type: SemanticType, cols: Iterable[str]):
        """
        Helper method to populate strong semantic detections. If the second parameter is non-empty
        then it is appended to the semantic detections mapping of the given type.

        :param semantic_type: The semantic type that should be mapped to the given column names
        :param cols: A (potentially empty) list of column names
        """
        if cols:
            self._strong_semantic_detections[semantic_type].extend(cols)

    def _add_weak_detections(self, semantic_type: SemanticType, cols: Iterable[str]):
        """
        Analogous to method to add strong detections.
        """
        if cols:
            self._weak_semantic_detections[semantic_type].extend(cols)

    def _is_compatible(self, col: str, semantic_type: str) -> bool:
        """
        Checks whether a given semantic type is compatible with the schema of a column. For the time
        being it employs the following compatibility matrix:

        [Semantic type --> Compatible physical types]
        CATEGORICAL --> string, numerics
        NUMERIC --> string, numerics
        DATETIME --> string, numerics, timestamps, dates
        TEXT --> string

        :param col: The column name
        :param semantic_type: The semantic type
        :return: Returns true if the given semantic type is compatible with the column's schema
            wrt supported AutoML's feature-engineering techniques.
        """
        compatible_types = defaultdict(
            list, {
                annotation_utils.CATEGORICAL: [StringType, self.NUMERICAL_TYPE],
                annotation_utils.NUMERIC: [StringType, self.NUMERICAL_TYPE],
                annotation_utils.DATETIME: [
                    StringType, self.NUMERICAL_TYPE, TimestampType, DateType
                ],
                annotation_utils.TEXT: [StringType],
            })
        compatible_columns = list(
            chain(*[self.feature_schema[t] for t in compatible_types[semantic_type]]))
        return semantic_type == annotation_utils.NATIVE or col in compatible_columns

    def detect_semantic_types(self, dataset: DataFrame, no_detection_cols: List[str]):
        """
        Detects semantic types for input columns with ambiguous data types.
        Uses a sample for large datasets.

        :param no_detection_cols: A list of column names that will not be considered for
            semantic type detection.
        """

        # Used to capture the annotations that are incompatible and will be ignored
        incompatible_annotations = defaultdict(list)

        for col in self._schema_info.supported_cols:
            if col == self._target_col or col in no_detection_cols:
                continue

            annotation = self._schema_info.feature_metadata[col].get(
                annotation_utils.SEMANTIC_TYPE_KEY, "")

            if not annotation:
                continue

            semantic_types = annotation if isinstance(annotation, list) else [annotation]
            for semantic_type in semantic_types:
                if self._is_compatible(col, semantic_type):
                    self._strong_semantic_detections[SemanticType(semantic_type)].append(col)
                else:
                    incompatible_annotations[col].append(semantic_type)

        string_cols = self._filter_cols_with_semantic_types(self.feature_schema[StringType])
        integer_cols = self._filter_cols_with_semantic_types([
            f.name for f in dataset.schema.fields
            if isinstance(f.dataType, self.INTEGER_TYPES) and f.name != self._target_col
        ])
        numeric_cols = self._filter_cols_with_semantic_types(
            self.feature_schema[self.NUMERICAL_TYPE])

        columns_to_detect = dataset.select(string_cols + numeric_cols)

        fraction = (self.TYPE_DETECTION_SAMPLE_SIZE * 1.1) / self._pre_sampling_stats.num_rows
        if fraction < 1.0:
            pdf = columns_to_detect.sample(
                fraction=fraction, seed=np.random.randint(1e9)).toPandas()
        else:
            pdf = columns_to_detect.toPandas()

        # categoricals
        cols_df = pdf[self._filter_cols_with_semantic_types(string_cols + numeric_cols)]
        categorical_cols = self._detect_strong_categoricals(cols_df)
        self._add_strong_detections(SemanticType.CATEGORICAL,
                                    [col for col in categorical_cols if col in numeric_cols])
        self._add_strong_detections(SemanticType.NATIVE,
                                    [col for col in categorical_cols if col in string_cols])

        # string -> datetime
        string_cols_df = pdf[self._filter_cols_with_semantic_types(string_cols)]
        datetime_string_cols = self._detect_semantic_type_by_conversion(
            string_cols_df, pd.to_datetime, errors="coerce", infer_datetime_format=True)
        self._add_strong_detections(SemanticType.DATETIME, datetime_string_cols)

        # string -> numeric
        string_cols_df = pdf[self._filter_cols_with_semantic_types(string_cols)]
        numeric_string_cols = self._detect_semantic_type_by_conversion(
            string_cols_df,
            self._convert_from_string_to_numeric,
        )
        self._add_strong_detections(SemanticType.NUMERIC, numeric_string_cols)

        # string -> text
        string_cols_df = pdf[self._filter_cols_with_semantic_types(string_cols)]
        text_cols = self._detect_english_text(string_cols_df)
        self._add_strong_detections(SemanticType.TEXT, text_cols)

        # integer -> datetime
        integer_cols_df = pdf[integer_cols]
        self._add_strong_detections(
            SemanticType.DATETIME,
            self._detect_semantic_type_by_conversion(
                integer_cols_df,
                self._convert_from_numeric_to_timestamp,
            ))

        # numeric -> categorical
        # Note: this may be a no-op given that strong categorical detection happens
        # at the beginning and weak detections are not currently forwarded. However, we leave
        # the code here so that we can independently tune numeric->categorical tuning without
        # impacting the initial detection across columns.
        remaining_numeric_cols = self._filter_cols_with_semantic_types(numeric_cols)
        numeric_cols_df = pdf[remaining_numeric_cols]
        strong_numeric_categoricals, weak_numeric_categoricals = self._detect_numeric_categoricals(
            numeric_cols_df)
        self._add_strong_detections(SemanticType.CATEGORICAL, strong_numeric_categoricals)
        self._add_weak_detections(SemanticType.CATEGORICAL, weak_numeric_categoricals)

        # Drop NATIVE detection prior to logging and return
        self._strong_semantic_detections.pop(SemanticType.NATIVE, None)

        # Drop no_detection_cols
        if no_detection_cols:
            # use list(...) to make a copy of the keys, so we can use del while iterating
            for semantic_type in list(self._strong_semantic_detections):
                new_cols = list(
                    set(self._strong_semantic_detections[semantic_type]) - set(no_detection_cols))
                if len(new_cols) == 0:
                    del self._strong_semantic_detections[semantic_type]
                else:
                    self._strong_semantic_detections[semantic_type] = new_cols

        # Add semantic detection to AutoML's alerts.
        self._feature_alert_handler.log_strong_semantic_detection_warnings(
            self._strong_semantic_detections)

        for semantic_type, columns in self._strong_semantic_detections.items():
            if columns:
                _logger.warning(create_semantic_type_message(columns, semantic_type))

        if incompatible_annotations:
            alert = IncompatibleSemanticTypeAnnotationAlert(incompatible_annotations)
            self._alert_manager.record(alert)
            for col, tpe in incompatible_annotations.items():
                tpe = ", ".join([f"`{t}`" for t in tpe])
                _logger.warning(f"Column '{col}' was annotated as semantic type {tpe}, " +
                                "but its type is incompatible, and the annotation will be ignored.")

    @staticmethod
    def _detect_strong_categoricals(cols_df: pd.DataFrame) -> List[str]:
        """
        Detects columns that can be treated as categorical due to their
        low cardinality (<= STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD).

        :param cols_df: A DataFrame with columns to be detected.
        :return: The list of column names that are strongly-detected as categorical.
        """
        cardinalities = cols_df.nunique()
        strong_detections = cols_df.columns[
            cardinalities <= SupervisedLearnerDataPreprocessor.
            STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD].tolist()

        return strong_detections

    @staticmethod
    def _detect_numeric_categoricals(numeric_cols_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Detects numeric columns that can be treated as categorical.
        Performs both strong and weak detection:

        - Strong detection: column cardinality <= STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD
        - Weak detection: column cardinality <= NUMERIC_CATEGORICAL_CARDINALITY_THRESHOLD

        :param numeric_cols_df: A DataFrame with numeric columns.
        :return: A tuple (s,w) of lists, where s is the list of column names that are
          strongly-detected as categorical and w is the list of column names that are
          weakly-detected as categorical.
        """
        cardinalities = numeric_cols_df.nunique()
        strong_detections = numeric_cols_df.columns[
            cardinalities <= SupervisedLearnerDataPreprocessor.
            STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD].tolist()
        weak_detections = numeric_cols_df.columns[cardinalities.between(
            SupervisedLearnerDataPreprocessor.STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD,
            SupervisedLearnerDataPreprocessor.NUMERIC_CATEGORICAL_CARDINALITY_THRESHOLD,
            inclusive="right")].tolist()

        return strong_detections, weak_detections

    @classmethod
    def _detect_english_text(cls, string_cols_df: pd.DataFrame) -> List[str]:
        """
        Detects string columns that can be treated as English text.

        We use the following criteria: if a string column has a percentage of distinct values
        > TEXT_PERCENT_DISTINCT_THRESHOLD, we concatenate all of the rows into a single string and
        use fasttext's language identification model to get an English confidence score.
        If the confidence is >= FASTTEXT_CONFIDENCE_THRESHOLD, we detect the column as English.
        For more details on this criteria, see
        https://docs.google.com/document/d/1F-vjTXcY2vg23ODrrZrBGNqObJt3jHeWOc2VpuM08dM.

        :param string_cols_df: A DataFrame with string columns.
        :return: list of column names that are strongly-detected as English text
        """
        # Don't load fasttext model if there are no string columns to detect.
        if len(string_cols_df.columns) == 0:
            return []

        ft_model = None
        try:
            with importlib.resources.path(
                    artifacts,
                    SupervisedLearnerDataPreprocessor.FASTTEXT_MODEL_FILE) as ft_model_path:

                with filter_stderr(
                        "Warning : `load_model` does not return WordVectorModel or SupervisedModel "
                        + "any more, but a `FastText` object which is very similar"):
                    ft_model = fasttext.load_model(str(ft_model_path))
        except:
            pass  # Do not fail if model cannot be loaded
        if not ft_model:
            _logger.warning(
                "AutoML was unable to load artifacts necessary for English text detection and "
                "will not attempt to detect which string columns represent text values.")
            return []

        detections = []
        for col in string_cols_df.columns:
            # Filter columns that have a low percentage of distinct values
            values = string_cols_df[col].dropna()
            if values.empty:
                continue
            percent_distinct = values.nunique() * 1.0 / values.size
            if percent_distinct <= cls.TEXT_PERCENT_DISTINCT_THRESHOLD:
                continue

            # Concatenate rows using space as separator; replace newlines for fasttext model
            text_concat = values.astype(str).str.cat(sep=" ").replace("\n", " ")

            # fasttext returns two tuples, corresponding to predicted languages and confidences
            languages, confidences = ft_model.predict(text_concat)
            lang_to_conf = dict(zip(languages, confidences))
            confidence = lang_to_conf.get(cls.FASTTEXT_ENGLISH_LABEL, -1)
            if confidence >= cls.FASTTEXT_CONFIDENCE_THRESHOLD:
                detections.append(col)
        return detections

    @staticmethod
    def _convert_from_numeric_to_timestamp(series: pd.Series) -> pd.Series:
        """
        Attempts to convert the input (numeric) series to a timestamp series. Returns a series of
        null values if not successful.

        Internally, the method attempts the conversion by serializing the input to JSON and then
        reading it back to a DataFrame. This takes advantage of Pandas' heuristics for timestamp
        detection.

        :param series: A numeric series of unique, non-null values
        :return: The series converted to a timestamp (if feasible) otherwise a series of np.nan.
        """
        # "table" orientation forces JSON serialization to include the column name.
        ORIENT = "table"
        # Not include index values in the JSON string. Otherwise there will be a conflict when existing a
        # column named as `index`.
        converted_series = pd.read_json(
            series.to_json(orient=ORIENT, index=False), orient=ORIENT)[series.name]
        failed_conversion = pd.Series(np.NaN, index=series.index)
        # Select only "datetime" columns. If the conversion succeeds then the output will have shape
        # (c, 1) where c is the number of input rows. Otherwise the result will have shape (c,0).
        if not pd_types.is_datetime64_ns_dtype(converted_series.dtype):
            return failed_conversion
        # Check if the original resolution is in nanoseconds. We do this by converting the converted
        # series to int and then comparing against the original series.
        converted_series_as_ns = converted_series.astype(int)

        return converted_series if (converted_series_as_ns == series).all() else failed_conversion

    @classmethod
    def _convert_from_string_to_numeric(cls, series: pd.Series) -> pd.Series:
        """
        Attempts to convert the input (string) series to a numeric series. Returns a series of
        null values if the converted series does not meet the criteria for strong numeric detection.
        Otherwise, will return the converted series.

        :param series: A string series of unique, non-null values
        :return: The series converted to numeric (if successful) otherwise a series of np.nan.
        """
        failed_conversion = pd.Series(np.NaN, index=series.index)

        # Check if cardinality is below categorical/numeric boundary threshold
        cardinality = series.size
        if cardinality <= cls.NUMERIC_CATEGORICAL_CARDINALITY_THRESHOLD:
            return failed_conversion

        converted_series = pd.to_numeric(series, errors="coerce")

        # Check if cardinality exceeds strong numeric threshold
        if cardinality >= cls.STRONG_NUMERIC_DETECTION_CARDINALITY_THRESHOLD:
            return converted_series

        # If result type is float, check if there are any non-zero fractional parts
        # np.modf will return a tuple of arrays: (fractionals, integrals)
        if pd_types.is_float_dtype(converted_series.dtype) and np.modf(converted_series)[0].any():
            return converted_series
        else:
            return failed_conversion

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

            if col in self.feature_schema[self.NUMERICAL_TYPE]:
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
               set(self.feature_schema[self.UNSUPPORTED_TYPE])) >= len(
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
            unsupported_columns=set(self.feature_schema[self.UNSUPPORTED_TYPE]) |
            unsupported_array_columns,
            constant_columns=constant_columns,
            date_columns=set(self.feature_schema[DateType]),
            timestamp_columns=timestamp_columns,
            num_nulls=num_nulls,
            target_col_type=self._target_col_type,
            strong_semantic_detections=self.strong_semantic_detections,
            size_estimator_result=size_estimator_result,
            num_classes=num_classes)

        return preprocess_result
