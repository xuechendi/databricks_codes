from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Set

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import approx_count_distinct, avg, col, count, isnan, length, when, countDistinct
from pyspark.sql.types import ArrayType, NumericType, StringType

from databricks.automl.legacy.confs import InternalConfs
from databricks.automl.legacy.const import AutoMLDataType, SemanticType
from databricks.automl.legacy.problem_type import ClassificationTargetTypes, ProblemType


@dataclass
class PreSamplingColumnStats:
    approx_num_distinct: int
    num_nulls: int
    str_avg_length: Optional[float] = None


@dataclass
class PreSamplingStats:
    num_rows: int
    num_invalid_rows: int
    num_target_nulls: int
    schema_map: Dict[str, Set[str]]
    str_columns: Set[str]
    class_counts: Optional[
        Dict[ClassificationTargetTypes, int]]  # only included for classification problems
    columns: Dict[str, PreSamplingColumnStats]


@dataclass
class ColumnStats:
    type: AutoMLDataType
    approx_num_distinct: int  # this is copied from PreSamplingColumnStats
    num_distinct: Optional[int]
    num_missing: int
    is_constant: Optional[bool]


@dataclass
class ArrayColumnStats(ColumnStats):
    elementType: AutoMLDataType
    max_length: int
    min_length: int


@dataclass
class PostSamplingStats:
    num_rows: int
    columns: Dict[str, ColumnStats]  # each key is the column name


class StatsCalculator:
    APPROX_NUM_DISTINCT_LIKELY_UNIQUE_THRESHOLD = 0.9

    # 5 rows in total are required when we adopt the sample-and-split, so that we can
    # split the sampled data into train/val/test data for a minimum of 3/1/1 rows.
    MIN_ROWS_PER_LABEL_BEFORE_SPLIT = 5
    # Only 1 row is required after sampling when we adopt the split-and-sample workflow.
    MIN_ROWS_PER_LABEL_AFTER_SPLIT = 1
    # Number of rows required by pyspark.sampleBy(..) to make sure we at least get
    # 1 or 5 row(s) when we sample the dataset.
    # See the below notebook, where sampleBy(..) is called 1000 times to ensure enough rows are
    # left after sampleBy(..):
    # http://go/dogfood/?o=6051921418418893#notebook/3603418870987052/command/3603418870987064
    PYSPARK_MIN_OUTPUT_ROWS_TO_INPUT_ROWS_MAP = {1: 10, 5: 20}

    # Minimum number of rows per class required for AutoML to successfully run test-train-validation split
    MIN_ROWS_PER_CLASS = int(PYSPARK_MIN_OUTPUT_ROWS_TO_INPUT_ROWS_MAP[MIN_ROWS_PER_LABEL_AFTER_SPLIT] / 0.6) \
        if InternalConfs.ENABLE_TRAIN_TEST_SPLIT_DRIVER \
        else MIN_ROWS_PER_LABEL_BEFORE_SPLIT

    @staticmethod
    def get_pre_sampling_stats(dataset: DataFrame, target_col: str,
                               problem_type: ProblemType) -> PreSamplingStats:
        """
        Make one pass on the pyspark dataset and calculate the total number or rows,
        average length of string columns, and the approximate cardinality of string columns
        Also include information about the dataset schema

        :param dataset: pyspark dataset
        :param target_col: name of the target column
        :param problem_type: problem type i.e. classification / regression / forecasting
        :return: PreSamplingStats
        """
        schema_map = defaultdict(set)
        for c, t in dataset.dtypes:
            schema_map[t].add(c)
        str_columns = schema_map.get("string", set())

        select_constructs = [
            count("*").alias("count"),
            count(when(F.col(target_col).isNull(), target_col)).alias("count_target_nulls")
        ]
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
        stats = dataset.select(select_constructs).collect()[0].asDict()

        columnStats = {}
        for col in dataset.columns:
            # Note: stats["xxx"] might return `None` if all values are NULL.
            columnStats[col] = PreSamplingColumnStats(
                approx_num_distinct=stats[f"approx_num_distinct_{col}"] or 1,
                num_nulls=stats[f"num_nulls_{col}"])
            if col in str_columns:
                columnStats[col].str_avg_length = stats[f"avg_length_{col}"] or 0

        class_counts = None
        num_low_class_count_rows = 0

        # For classification, calculate the class counts and also calculate the number of
        # rows that have less than the min required class counts
        if problem_type == ProblemType.CLASSIFICATION:
            class_counts = StatsCalculator.get_class_counts(dataset, target_col)

            num_low_class_count_rows = sum([
                ct for c, ct in class_counts.items()
                if ct < StatsCalculator.MIN_ROWS_PER_CLASS and c is not None
            ])

        return PreSamplingStats(
            num_rows=stats["count"],
            num_invalid_rows=stats["count_target_nulls"] + num_low_class_count_rows,
            num_target_nulls=stats["count_target_nulls"],
            schema_map=dict(schema_map),
            str_columns=str_columns,
            class_counts=class_counts,
            columns=columnStats)

    @staticmethod
    def get_likely_unique_columns(pre_sampling_stats: PreSamplingStats) -> List[str]:
        """
        Returns a list of columns whose approx_count_distinct is sufficiently close
        to the number of rows in the dataset.

        approx_count_distinct has a default standard deviation of 0.05
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.approx_count_distinct.html
        """
        likely_unique_columns = []
        for col_name, col_stats in pre_sampling_stats.columns.items():
            if col_stats.approx_num_distinct > StatsCalculator.APPROX_NUM_DISTINCT_LIKELY_UNIQUE_THRESHOLD * pre_sampling_stats.num_rows:
                likely_unique_columns.append(col_name)
        return likely_unique_columns

    @staticmethod
    def get_class_counts(dataset: DataFrame,
                         target_col: str) -> Dict[ClassificationTargetTypes, int]:
        """
        Get the number of rows for each class in a classification dataset
        When the cardinality of the target column is high, this query is very slow!
        :return: dictionary where each key is a class and each value is the number of rows of that class
        """
        class_counts_df = dataset.groupBy(target_col).count()
        class_counts = {row[target_col]: row["count"] for row in class_counts_df.collect()}
        return class_counts

    @staticmethod
    def get_post_sampling_stats(dataset: DataFrame,
                                target_col: str,
                                problem_type: ProblemType,
                                strong_semantic_detections: Dict[SemanticType, List[str]],
                                pre_sampling_stats: PreSamplingStats,
                                exclude_cols: List[str] = []) -> PostSamplingStats:
        likely_unique_columns = StatsCalculator.get_likely_unique_columns(pre_sampling_stats)
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

            if isinstance(dataset.schema[c].dataType, ArrayType):
                select_constructs.append(F.max(F.size(c)).alias(f"max_length_{c}"))
                select_constructs.append(F.min(F.size(c)).alias(f"min_length_{c}"))
            else:
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
                    max_length=stats[f"max_length_{name}"],
                    min_length=stats[f"min_length_{name}"],
                )
            elif name not in exclude_cols:
                columns[name] = ColumnStats(
                    type=type_,
                    approx_num_distinct=pre_sampling_stats.columns[name].approx_num_distinct,
                    num_distinct=stats.get(f"distinct_{name}", None),
                    num_missing=stats[f"nulls_{name}"],
                    is_constant=stats[f"is_constant_{name}"],
                )

        return PostSamplingStats(
            num_rows=stats["num_rows"],
            columns=columns,
        )
