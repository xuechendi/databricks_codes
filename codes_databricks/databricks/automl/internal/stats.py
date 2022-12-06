import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple, Type, Union

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.functions import approx_count_distinct, avg, col, count, isnan, length, when, countDistinct
from pyspark.sql.types import ArrayType, DataType, NumericType, StringType, StructType

from databricks.automl.shared.const import ClassificationTargetTypes
from databricks.automl.internal.alerts import NoFeatureColumnsAlert, UnsupportedColumnAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.common.const import AutoMLDataType, SparkDataType
from databricks.automl.shared.errors import UnsupportedDataError
from databricks.automl.shared.const import SemanticType, ProblemType

_logger = logging.getLogger(__name__)


@dataclass
class InputStats:
    """ The stats are used for logging and the rough sampling step. """
    num_rows: int
    num_invalid_rows: int
    num_target_nulls: int
    num_cols: int
    num_string_cols: int
    num_supported_cols: int


@dataclass
class IntermediateColumnStats:
    approx_num_distinct: int
    num_nulls: int
    str_avg_length: Optional[float] = None  # Only for string column
    array_min_length: Optional[float] = None  # Only for array column
    array_max_length: Optional[float] = None  # Only for array column


@dataclass
class IntermediateStats:
    # Only re-calculated if new features joined
    num_rows: int
    schema_map: Dict[str, Set[str]]
    str_columns: Set[str]
    columns: Dict[str, IntermediateColumnStats]
    # Mapping from types to column names
    feature_schema: Mapping[Union[str, Type[DataType]], List[str]]
    # Metadata for each column name
    feature_metadata: Mapping[str, Dict]
    # Supported column names
    supported_cols: List[str]
    # Always 0 because this is calculated after dropping invalid rows. Kept for data process and sample.
    num_invalid_rows: int


@dataclass
class ColumnStats:
    type: AutoMLDataType
    approx_num_distinct: int  # this is copied from IntermediateColumnStats
    num_distinct: Optional[int]
    num_missing: int
    is_constant: Optional[bool]


@dataclass
class ArrayColumnStats(ColumnStats):
    elementType: AutoMLDataType
    max_length: int  # this is copied from IntermediateColumnStats
    min_length: int  # this is copied from IntermediateColumnStats


@dataclass
class PostSamplingStats:
    num_rows: int
    columns: Dict[str, ColumnStats]  # each key is the column name


@dataclass
class SchemaInfo:
    # Mapping from types to column names
    feature_schema: Mapping[Union[str, Type[DataType]], List[str]]
    # Metadata for each column name
    feature_metadata: Mapping[str, Dict]
    # Supported column names
    supported_cols: List[str]


class StatsCalculator:
    APPROX_NUM_DISTINCT_LIKELY_UNIQUE_THRESHOLD = 0.9

    def __init__(self, target_col: str, time_col: Optional[str], run_id: str):
        self._target_col = target_col
        self._time_col = time_col
        self._run_id = run_id

    def _get_schema_info(self, dataset_schema: StructType) -> SchemaInfo:
        """
        Unpacks schema information in the input dataset.
        :param target_col: name of target column
        :return: A FeatureSchema data class
        """
        feature_schema = defaultdict(list)
        supported_cols = []
        fields = [field for field in dataset_schema.fields if field.name != self._target_col]
        feature_metadata = dict()
        for field in fields:
            if isinstance(field.dataType, SparkDataType.ALL_TYPES):
                # condense feature schema into format {dtype: [column_names]}
                # collapse numerical types into one
                supported_cols.append(field.name)
                if isinstance(field.dataType, SparkDataType.NUMERIC_TYPES):
                    feature_schema[SparkDataType.NUMERICAL_TYPE].append(field.name)
                else:
                    feature_schema[type(field.dataType)].append(field.name)
            else:
                feature_schema[SparkDataType.UNSUPPORTED_TYPE].append(field.name)
            feature_metadata[field.name] = field.metadata
        supported_cols.append(self._target_col)

        return feature_schema, supported_cols, feature_metadata

    def get_input_stats(self, dataset: DataFrame) -> InputStats:
        """
        Calculate the stats used for InputDataStats logging and the rough sampling step.

        :param dataset: pyspark dataset
        :return: InputStats
        """
        schema_map, str_columns = StatsCalculator._get_schema_and_str_columns(dataset)

        select_constructs = [
            count("*").alias("count"),
            count(when(F.col(self._target_col).isNull(),
                       self._target_col)).alias("count_target_nulls")
        ]
        stats = dataset.select(select_constructs).collect()[0].asDict()

        _, supported_cols, _ = self._get_schema_info(dataset_schema=dataset.schema)

        input_stats = InputStats(
            num_rows=stats["count"],
            num_invalid_rows=stats["count_target_nulls"],
            num_target_nulls=stats["count_target_nulls"],
            num_cols=len(dataset.columns),
            num_string_cols=len(str_columns),
            num_supported_cols=len(supported_cols))
        _logger.debug(f"InputStats calculated: {input_stats}")

        return input_stats

    @staticmethod
    def _get_schema_and_str_columns(dataset: DataFrame):
        schema_map = defaultdict(set)
        for c, t in dataset.dtypes:
            schema_map[t].add(c)
        return schema_map, schema_map.get("string", set())

    def get_intermediate_stats(self,
                               dataset: DataFrame,
                               input_num_rows: int,
                               is_sampled: bool = False) -> IntermediateStats:
        """
        Make one pass on the pyspark dataset and calculate the total number of rows,
        average length of string columns, and the approximate cardinality of string columns
        Also include information about the dataset schema

        :param dataset: pyspark dataset
        :return: IntermediateStats
        """
        schema_map, str_columns = StatsCalculator._get_schema_and_str_columns(dataset)

        select_constructs = [count("*").alias("count")] if is_sampled else []
        for col in dataset.columns:
            select_constructs.append(approx_count_distinct(col).alias(f"approx_num_distinct_{col}"))
            if isinstance(dataset.schema[col].dataType, NumericType):
                select_constructs.append(
                    count(when(F.col(col).isNull() | F.isnan(col), col)).alias(f"num_nulls_{col}"))
            else:
                select_constructs.append(
                    count(when(F.col(col).isNull(), col)).alias(f"num_nulls_{col}"))
            if col in str_columns:
                select_constructs.append(avg(length(col)).alias(f"avg_length_{col}"))
            if isinstance(dataset.schema[col].dataType, ArrayType):
                select_constructs.append(F.max(F.size(col)).alias(f"max_length_{col}"))
                select_constructs.append(F.min(F.size(col)).alias(f"min_length_{col}"))
        stats = dataset.select(select_constructs).collect()[0].asDict()

        columnStats = {}
        for col in dataset.columns:
            # Note: stats["xxx"] might return `None` if all values are NULL.
            columnStats[col] = IntermediateColumnStats(
                approx_num_distinct=stats[f"approx_num_distinct_{col}"] or 1,
                num_nulls=stats[f"num_nulls_{col}"])
            if col in str_columns:
                columnStats[col].str_avg_length = stats[f"avg_length_{col}"] or 0
            if isinstance(dataset.schema[col].dataType, ArrayType):
                columnStats[col].array_min_length = stats[f"min_length_{col}"] or 0
                columnStats[col].array_max_length = stats[f"max_length_{col}"] or 0

        # Unpacks schema information in the input dataset.
        feature_schema, supported_cols, feature_metadata = self._get_schema_info(
            dataset_schema=dataset.schema)

        input_stats = IntermediateStats(
            num_rows=stats["count"] if is_sampled else input_num_rows,
            schema_map=dict(schema_map),
            str_columns=str_columns,
            columns=columnStats,
            feature_schema=feature_schema,
            supported_cols=supported_cols,
            feature_metadata=feature_metadata,
            num_invalid_rows=0)
        _logger.debug(f"DataStats calculated before precise sampling: {input_stats}")

        return input_stats

    def validate_intermediate_stats(
            self,
            input_stats: IntermediateStats,
            alert_manager: AlertManager) -> \
            Tuple[bool, Optional[List[float]], Optional[ClassificationTargetTypes], DataFrame]:
        """
        Validate the pre-sampling stats.
        :param input_stats: Result for pre sampling.
        :param alert_manager: AlertManager used to pass warnings to the user.
        :return: indicator whether we should balance data, list of target_label_ratios, and
            processed positive label
        """
        self._validate_unsupported_columns(input_stats, alert_manager)
        return False, None, None

    def _validate_unsupported_columns(self, input_stats: IntermediateStats,
                                      alert_manager: AlertManager):
        # Check and alert the unsupported data types
        unsupported_cols = input_stats.feature_schema.get(SparkDataType.UNSUPPORTED_TYPE, [])
        if unsupported_cols:
            alert_manager.record(UnsupportedColumnAlert(unsupported_cols))
            _logger.warning(
                f"The following columns are of unsupported data types and will be dropped: {unsupported_cols}"
            )

        # calculate supported cols that have all nulls
        num_rows = input_stats.num_rows
        empty_supported_cols = [
            col for col in input_stats.supported_cols
            if input_stats.columns[col].num_nulls == num_rows
        ]
        if len(empty_supported_cols) > 0:
            # only log to console, these will be detected as constant
            # columns later on and dropped in training
            _logger.info(f"Following columns are found to have all nulls "
                         f"and will be dropped: {empty_supported_cols}")

        supported_non_empty_cols = set(input_stats.supported_cols) - \
                                   set(empty_supported_cols) - set([self._target_col])
        if len(supported_non_empty_cols) == 0:
            alert_manager.record(NoFeatureColumnsAlert())
            raise UnsupportedDataError("No supported column types found in dataset")

    @staticmethod
    def _get_likely_unique_columns(input_stats: IntermediateStats) -> List[str]:
        """
        Returns a list of columns whose approx_count_distinct is sufficiently close
        to the number of rows in the dataset.

        approx_count_distinct has a default standard deviation of 0.05
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.approx_count_distinct.html
        """
        likely_unique_columns = []
        for col_name, col_stats in input_stats.columns.items():
            if col_stats.approx_num_distinct > StatsCalculator.APPROX_NUM_DISTINCT_LIKELY_UNIQUE_THRESHOLD * input_stats.num_rows:
                likely_unique_columns.append(col_name)
        return likely_unique_columns

    @staticmethod
    def get_post_sampling_stats(dataset: DataFrame,
                                target_col: str,
                                problem_type: ProblemType,
                                strong_semantic_detections: Dict[SemanticType, List[str]],
                                intermediate_stats: IntermediateStats,
                                exclude_cols: List[str] = []) -> PostSamplingStats:
        likely_unique_columns = StatsCalculator._get_likely_unique_columns(intermediate_stats)
        col_to_semantic_type = {}
        for semantic_type, columns in strong_semantic_detections.items():
            for column in columns:
                col_to_semantic_type[column] = semantic_type

        col_to_automl_type = {}
        for name in dataset.columns:
            if name in col_to_semantic_type:
                type_ = AutoMLDataType.from_semantic_type(col_to_semantic_type[name])
            else:
                type_ = AutoMLDataType.from_spark_type(dataset.schema[name].dataType)
            col_to_automl_type[name] = type_

        first_value = dataset.first()
        select_constructs = [count("*").alias("num_rows")]
        for c in dataset.columns:
            # countDistinct is expensive, so do it only when necessary
            if (isinstance(dataset.schema[c].dataType, StringType) and
                    c in likely_unique_columns) or (problem_type == ProblemType.CLASSIFICATION and
                                                    c == target_col):
                select_constructs.append(countDistinct(col(c)).alias(f"distinct_{c}"))

            if not isinstance(dataset.schema[c].dataType, ArrayType):
                # create is_constant_{c} for all non-array columns
                if first_value[c] is None or (isinstance(first_value[c], float) and
                                              math.isnan(first_value[c])):
                    # column is constant if all entries are null or nan
                    if isinstance(dataset.schema[c].dataType, NumericType):
                        select_constructs.append(
                            (count(when(col(c).isNull() | isnan(c),
                                        c)) == count("*")).alias(f"is_constant_{c}"))
                    else:
                        select_constructs.append((count(when(
                            col(c).isNull(), c)) == count("*")).alias(f"is_constant_{c}"))
                else:
                    # column is constant if all entries equal the first entry
                    # this method doesn't work if first_value[c] is None or nan, so we have the `if` condition workaround
                    select_constructs.append((count(when(col(c) != first_value[c],
                                                         c)) == 0).alias(f"is_constant_{c}"))

            if isinstance(dataset.schema[c].dataType, NumericType):
                select_constructs.append(
                    count(when(col(c).isNull() | isnan(c), c)).alias(f"nulls_{c}"))
            else:
                select_constructs.append(count(when(col(c).isNull(), c)).alias(f"nulls_{c}"))
        stats = dataset.select(select_constructs).collect()[0].asDict()

        columns = {}
        for name, type_ in col_to_automl_type.items():
            if type_ == AutoMLDataType.ARRAY:
                columns[name] = ArrayColumnStats(
                    type=type_,
                    approx_num_distinct=None,
                    num_distinct=None,
                    num_missing=stats[f"nulls_{name}"],
                    is_constant=None,
                    elementType=AutoMLDataType.from_spark_type(
                        dataset.schema[name].dataType.elementType),
                    max_length=intermediate_stats.columns[name].array_max_length,
                    min_length=intermediate_stats.columns[name].array_min_length,
                )
            elif name not in exclude_cols:
                columns[name] = ColumnStats(
                    type=type_,
                    approx_num_distinct=intermediate_stats.columns[name].approx_num_distinct,
                    num_distinct=stats.get(f"distinct_{name}", None),
                    num_missing=stats[f"nulls_{name}"],
                    is_constant=stats[f"is_constant_{name}"],
                )

        return PostSamplingStats(
            num_rows=stats["num_rows"],
            columns=columns,
        )
