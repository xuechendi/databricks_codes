import importlib
import logging
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Tuple, Iterable

import fasttext
import numpy as np
import pandas as pd
import pandas.api.types as pd_types
from pyspark.databricks.sql import annotation_utils
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, StringType, DateType, StructType, TimestampType

from databricks.automl.internal import artifacts
from databricks.automl.internal.alerts import NoFeatureColumnsAlert, SingleClassInTargetColumnAlert, \
    IncompatibleSemanticTypeAnnotationAlert
from databricks.automl.internal.alerts.alert_handler import FeatureAlertsHandler
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.alerts.feature_alert import *
from databricks.automl.internal.common.const import AutoMLDataType, SparseOrDense, SparkDataType
from databricks.automl.internal.utils.io_utils import filter_stderr
from databricks.automl.internal.stats import IntermediateStats, PostSamplingStats
from databricks.automl.shared.const import SemanticType

_logger = logging.getLogger(__name__)


class SemanticDetector:
    """
    Making Semantic detections for input dataset

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

    def __init__(self, intermediate_stats: IntermediateStats, target_col: str,
                 alert_manager: AlertManager):

        # initialize helper module to generate alerts
        self._feature_alert_handler = FeatureAlertsHandler(alert_manager)
        self._alert_manager = alert_manager

        self._num_rows = intermediate_stats.num_rows
        self._feature_metadata = intermediate_stats.feature_metadata
        self._feature_schema = intermediate_stats.feature_schema
        self._supported_cols = intermediate_stats.supported_cols
        self._target_col = target_col

        self._strong_semantic_detections = defaultdict(list)  # Maps semantic types to column names
        self._weak_semantic_detections = defaultdict(list)  # Maps semantic types to column names

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

    def _add_strong_detections(self, semantic_type: SemanticType, cols: Iterable[str]) -> None:
        """
        Helper method to populate strong semantic detections. If the second parameter is non-empty
        then it is appended to the semantic detections mapping of the given type.

        :param semantic_type: The semantic type that should be mapped to the given column names
        :param cols: A (potentially empty) list of column names
        """
        if cols:
            self._strong_semantic_detections[semantic_type].extend(cols)

    def _add_weak_detections(self, semantic_type: SemanticType, cols: Iterable[str]) -> None:
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
                annotation_utils.CATEGORICAL: [StringType, SparkDataType.NUMERICAL_TYPE],
                annotation_utils.NUMERIC: [StringType, SparkDataType.NUMERICAL_TYPE],
                annotation_utils.DATETIME: [
                    StringType, SparkDataType.NUMERICAL_TYPE, TimestampType, DateType
                ],
                annotation_utils.TEXT: [StringType],
            })
        compatible_columns = list(
            chain(*[self._feature_schema[t] for t in compatible_types[semantic_type]]))
        return semantic_type == annotation_utils.NATIVE or col in compatible_columns

    def detect_semantic_types(
            self, dataset: DataFrame, no_detection_cols: List[str]
    ) -> Tuple[Dict[SemanticType, List[str]], Dict[SemanticType, List[str]]]:
        """
        Detects semantic types for input columns with ambiguous data types.
        Uses a sample for large datasets.

        :param no_detection_cols: A list of column names that will not be considered for
            semantic type detection.
        """
        try:
            return self._detect_semantic_types_impl(dataset, no_detection_cols)
        except Exception as e:
            _logger.warning(f"Semantic detection failed with error {repr(e)}.\n"
                            f"AutoML will use the default column types")
            return self._strong_semantic_detections, self._weak_semantic_detections

    def _detect_semantic_types_impl(
            self, dataset: DataFrame, no_detection_cols: List[str]
    ) -> Tuple[Dict[SemanticType, List[str]], Dict[SemanticType, List[str]]]:

        # Used to capture the annotations that are incompatible and will be ignored
        incompatible_annotations = defaultdict(list)

        for col in self._supported_cols:
            if col == self._target_col or col in no_detection_cols:
                continue

            annotation = self._feature_metadata[col].get(annotation_utils.SEMANTIC_TYPE_KEY, "")

            if not annotation:
                continue

            semantic_types = annotation if isinstance(annotation, list) else [annotation]
            for semantic_type in semantic_types:
                if self._is_compatible(col, semantic_type):
                    self._strong_semantic_detections[SemanticType(semantic_type)].append(col)
                else:
                    incompatible_annotations[col].append(semantic_type)

        string_cols = self._filter_cols_with_semantic_types(self._feature_schema[StringType])
        integer_cols = self._filter_cols_with_semantic_types([
            f.name for f in dataset.schema.fields
            if isinstance(f.dataType, SparkDataType.INTEGER_TYPES) and f.name != self._target_col
        ])
        numeric_cols = self._filter_cols_with_semantic_types(
            self._feature_schema[SparkDataType.NUMERICAL_TYPE])

        columns_to_detect = dataset.select(string_cols + numeric_cols)

        fraction = (self.TYPE_DETECTION_SAMPLE_SIZE * 1.1) / self._num_rows
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

        _logger.debug(f"Weak semantic detections: {self._weak_semantic_detections}")
        _logger.debug(f"Strong semantic detections: {self.strong_semantic_detections}")
        return self.strong_semantic_detections, self._weak_semantic_detections

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
            cardinalities <=
            SemanticDetector.STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD].tolist()

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
            cardinalities <=
            SemanticDetector.STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD].tolist()
        weak_detections = numeric_cols_df.columns[cardinalities.between(
            SemanticDetector.STRONG_CATEGORICAL_DETECTION_CARDINALITY_THRESHOLD,
            SemanticDetector.NUMERIC_CATEGORICAL_CARDINALITY_THRESHOLD,
            inclusive="right")].tolist()

        return strong_detections, weak_detections

    @staticmethod
    def _detect_english_text(string_cols_df: pd.DataFrame) -> List[str]:
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
            with importlib.resources.path(artifacts,
                                          SemanticDetector.FASTTEXT_MODEL_FILE) as ft_model_path:

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
            if percent_distinct <= SemanticDetector.TEXT_PERCENT_DISTINCT_THRESHOLD:
                continue

            # Concatenate rows using space as separator; replace newlines for fasttext model
            text_concat = values.astype(str).str.cat(sep=" ").replace("\n", " ")

            # fasttext returns two tuples, corresponding to predicted languages and confidences
            languages, confidences = ft_model.predict(text_concat)
            lang_to_conf = dict(zip(languages, confidences))
            confidence = lang_to_conf.get(SemanticDetector.FASTTEXT_ENGLISH_LABEL, -1)
            if confidence >= SemanticDetector.FASTTEXT_CONFIDENCE_THRESHOLD:
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

    @staticmethod
    def _convert_from_string_to_numeric(series: pd.Series) -> pd.Series:
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
        if cardinality <= SemanticDetector.NUMERIC_CATEGORICAL_CARDINALITY_THRESHOLD:
            return failed_conversion

        converted_series = pd.to_numeric(series, errors="coerce")

        # Check if cardinality exceeds strong numeric threshold
        if cardinality >= SemanticDetector.STRONG_NUMERIC_DETECTION_CARDINALITY_THRESHOLD:
            return converted_series

        # If result type is float, check if there are any non-zero fractional parts
        # np.modf will return a tuple of arrays: (fractionals, integrals)
        if pd_types.is_float_dtype(converted_series.dtype) and np.modf(converted_series)[0].any():
            return converted_series
        else:
            return failed_conversion
