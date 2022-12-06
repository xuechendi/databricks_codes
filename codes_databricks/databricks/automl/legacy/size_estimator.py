import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Union, Iterable, Optional

import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, FloatType, LongType, IntegerType, \
    BooleanType, ShortType, ByteType, DateType, TimestampType

from databricks.automl.legacy.confs import InternalConfs
from databricks.automl.legacy.const import SemanticType, SparseOrDense
from databricks.automl.legacy.preprocess import SupervisedLearnerDataPreprocessConfs, SizeEstimatorResult
from databricks.automl.legacy.stats import PreSamplingStats
from databricks.automl.legacy.sections.training.preprocess import TextPreprocessor

_logger = logging.getLogger(__name__)

BYTES_IN_MiB = 1024**2
BYTES_IN_GiB = 1024**3


class MemoryEstimator:
    """A helper class to estimate the memory in a spark cluster."""

    @classmethod
    def get_memory(cls):
        """Get the best estimate of memory of the **current** instance (that runs the function)."""
        try:
            return cls._get_container_memory_bytes()
        except Exception as e:
            _logger.warning(f"Unable to get container memory: {e}")

        return cls._get_available_memory_bytes()

    @classmethod
    def _get_container_memory_bytes(cls):
        """Get the container memory, which is available from SPARK_WORKER_MEMORY environment variable.

        The environment variable is populated in
        https://github.com/databricks/universe/blob/master/cluster-common/src/main/scala/com/databricks/backend/cluster/SparkEnvConfConstants.scala
        and should be available for all Databricks clusters.

        The unit of the memory size can be 'm', 'mb', 'g', 'gb', as specified in
        https://spark.apache.org/docs/latest/configuration.html#spark-properties.

        This function will throw exception if container memory cannot be obtained.
        """
        import os
        env_var = os.environ["SPARK_WORKER_MEMORY"]
        if env_var.endswith("m"):
            return int(env_var[:-1]) * BYTES_IN_MiB
        elif env_var.endswith("mb"):
            return int(env_var[:-2]) * BYTES_IN_MiB
        elif env_var.endswith("g"):
            return int(env_var[:-1]) * BYTES_IN_GiB
        elif env_var.endswith("gb"):
            return int(env_var[:-2]) * BYTES_IN_GiB
        else:
            raise ValueError(f"Invalid envirionment variable SPARK_WORKER_MEMORY: {env_var}")

    @classmethod
    def _get_available_memory_bytes(cls):
        """Get the available memory returned in the instance."""
        import psutil
        return psutil.virtual_memory().total

    @classmethod
    def get_worker_memory(cls, sparkContext):
        """Runs a spark job to the the best estimate of memory on workers."""

        def _get_memory(context):
            return [cls.get_memory()]

        return sparkContext.parallelize(range(1), 1).mapPartitions(_get_memory).collect()[0]


class SizeEstimator:
    def __init__(self, spark):
        self.spark = spark
        self._pd_size_map = self._get_pd_size_map()
        self._num_cores_per_trial = int(spark.conf.get("spark.task.cpus", "1"))
        self._memory_mb_per_trial = self._get_available_memory_mb_per_trial()

    # keys are pandas equivalent pyspark types using:
    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.dtypes.html
    _POST_FE_COLUMN_MAPPING = {
        "double": 1,
        "float": 1,
        "bigint": 1,
        "int": 1,
        "smallint": 1,
        "tinyint": 1,
        "boolean": 2,
        "timestamp": 36,  # 13 -1 + 24(ohe)
        "date": 10
    }

    # Estimated memory = _CONTAINER_MEMORY_MB_MULTIPLIER * total container memory
    _CONTAINER_MEMORY_MB_MULTIPLIER = 0.7

    _PYSPARK_TYPE_FLOAT = "float"
    _PYSPARK_TYPE_STRING = "string"
    _PYSPARK_TYPE_TIMESTAMP = "timestamp"

    _PANDAS_MEMORY_USAGE_FACTOR = 2  # Memory scale-up required for .toPandas()
    _TEST_TRAIN_SPLIT_FACTOR = 0.6  # Fraction of dataset used for training
    _MIN_MEMORY_MB_AVAILABLE_PER_TRIAL = 1500  # Minimum memory to assume is available per trial to ensure memory estimation doesn't undershoot
    _MAX_TRAINING_SCALE_UP_FACTOR = 5  # Memory scale-up required for running training

    _BYTES_IN_NUMPY_ARRAY_ELEMENT = 8  # To convert from float64 (double) to bytes

    # scipy.sparse use int32 for index, while we assume the values to be float64.
    # Therefore if the matrix needs N byte to store the values, it needs a 50% overhead to store the indices.
    _SPARSE_MATRIX_INDEX_OVERHEAD = 1.5

    # This factor accounts for potential computation overhead for sparse matrix over the dense one.
    # For example, computing matrix-vector multiplication for a sparse matrix with ~50% non-zeros
    # (when overhead == 1) will be more expensive for doing the same calculation in the dense format.
    # We set the overhead to 2.0 as a first-order guestimate, while a more careful benchmark could yield
    # a more accurate number.
    _SPARSE_COMPUTATION_OVERHEAD = 2.0

    def _get_pd_size_map(self):
        # create a schema with nullable=True and create a dataframe with nulls
        # to ensure that the right pd datatypes are initialized
        schema = StructType([
            StructField("double", DoubleType(), True),
            StructField("float", FloatType(), True),
            StructField("bigint", LongType(), True),
            StructField("int", IntegerType(), True),
            StructField("smallint", ShortType(), True),
            StructField("tinyint", ByteType(), True),
            StructField("boolean", BooleanType(), True),
            StructField("string", StringType(), True),
            StructField("date", DateType(), True),
            StructField("timestamp", TimestampType(), True),
            # type does not matter, all nulls are converted to pd.object that take up 16 bytes
            StructField("unsupported", BooleanType(), True)
        ])

        pdf = self.spark.createDataFrame(
            data=[(10.0, 10.1, 1, 1, 1, 1, True, "", datetime.now(), datetime.now(), None),
                  (None, None, None, None, None, None, None, None, None, None, None)],
            schema=schema).toPandas()

        pdf["date"] = pd.to_datetime(pdf["date"])
        return {c: sys.getsizeof(pdf.iloc[0][c]) for c in pdf.columns}

    def _probe_for_num_cores_per_worker(self):
        def count_cpu(context):
            import os
            cpu_count = os.cpu_count(
            ) or 1  # os.cpu_count() may return None if it cannot determine the number of CPU cores
            return [cpu_count]

        return self.spark.sparkContext.parallelize(range(1),
                                                   1).mapPartitions(count_cpu).collect()[0]

    def _get_available_memory_mb_per_trial(self):
        num_cores_per_worker = self._probe_for_num_cores_per_worker()
        trials_per_worker = max(num_cores_per_worker // self._num_cores_per_trial, 1)
        if num_cores_per_worker // self._num_cores_per_trial < 1:
            _logger.warning("""The number of CPU cores requested per AutoML trial exceeds 
                            the number of CPU cores available per machine. Ensure the 
                            `spark.task.cpus` Spark configuration is set correctly.""")
        worker_memory_mb = MemoryEstimator.get_worker_memory(
            self.spark.sparkContext) / BYTES_IN_MiB * self._CONTAINER_MEMORY_MB_MULTIPLIER
        available_memory_mb_per_trial = worker_memory_mb / trials_per_worker
        _logger.debug(f"""
            num_cores_per_worker = {num_cores_per_worker}
            worker_memory_mb = {worker_memory_mb}
            num_cores_per_trial = {self._num_cores_per_trial}
            trials_per_worker = {trials_per_worker}
            available_memory_mb_per_trial = {available_memory_mb_per_trial}""")
        return max(available_memory_mb_per_trial, self._MIN_MEMORY_MB_AVAILABLE_PER_TRIAL)

    def _get_data_load_mem_req(self, stats: PreSamplingStats) -> float:
        """
        Calculate the maximum memory required by the dataset when loaded into memory and converted to pandas

        :param stats: Dataset stats for the pyspark dataset
        :return: Estimated memory in MB required to load this dataset into memory
        """
        row_size = 0
        for col_type, cols in stats.schema_map.items():
            row_size += len(cols) * (self._pd_size_map.get(col_type,
                                                           self._pd_size_map["unsupported"]))

        for col_name in stats.str_columns:
            # string takes up additional size depending upon the length of the string
            row_size += stats.columns[col_name].str_avg_length

        num_rows_to_sample_from = stats.num_rows - stats.num_invalid_rows
        mem_required = row_size * num_rows_to_sample_from * self._PANDAS_MEMORY_USAGE_FACTOR / BYTES_IN_MiB
        return mem_required

    def _get_num_values_in_col(self, col_name: str, col_type: Union[str, SemanticType],
                               stats: PreSamplingStats, sparse_or_dense: SparseOrDense) -> int:
        """
        Get number of values (features) a column will map to.

        :param col_name: the name of the column
        :param col_type: the type of the column
        :param stats: the pre-sampling stats, containing information like number of unique strings
        :param sparse_or_dense: whether the data is encoded as sparse or dense matrix
        :return: the number of features will be used by the model
        """
        if col_type == SemanticType.DATETIME:
            return self._POST_FE_COLUMN_MAPPING.get(self._PYSPARK_TYPE_TIMESTAMP)
        elif col_type == SemanticType.TEXT:
            return TextPreprocessor.DEFAULT_NUM_OUTPUT_COLS
        elif col_type == SemanticType.NUMERIC:
            return self._POST_FE_COLUMN_MAPPING.get(self._PYSPARK_TYPE_FLOAT)
        elif col_type == SemanticType.CATEGORICAL:
            if sparse_or_dense == SparseOrDense.SPARSE:
                return 1  # Only 1 non-zero for categorical column
            else:
                return stats.columns[col_name].approx_num_distinct
        elif col_type == self._PYSPARK_TYPE_STRING:
            threshold = SupervisedLearnerDataPreprocessConfs.DEFAULT_CATEGORICAL_EXTREME_CARDINALITY_THRESHOLD
            approx_num_distinct = stats.columns[col_name].approx_num_distinct
            if approx_num_distinct < threshold:
                if sparse_or_dense == SparseOrDense.SPARSE:
                    return 1  # Only 1 non-zero for categorical column
                else:
                    return min(
                        approx_num_distinct, SupervisedLearnerDataPreprocessConfs.
                        DEFAULT_CATEGORICAL_HIGH_CARDINALITY_THRESHOLD)
            else:
                # The string column with extreme cardinality is dropped
                return 0
        else:
            return self._POST_FE_COLUMN_MAPPING.get(col_type, 0)

    def _get_training_mem_req(self, stats: PreSamplingStats,
                              strong_semantic_detections: Dict[SemanticType, List[str]],
                              sparse_or_dense: SparseOrDense) -> float:
        """
        Calculate the maximum memory required when training this dataset, taking into account the feature
        engineering applied to the dataset in the training pipeline

        :param stats: Dataset stats for the pyspark dataset
        :param strong_semantic_detections: Dictionary with strongly detected semantic types and their column names
        :return: Estimated memory in MB required to train this dataset
        """

        # Construct a map from cols to their types, taking into account the semantic type detection
        # which might override the default column type.
        def pivot_to_col(types_to_cols: Dict[Union[str, SemanticType], Iterable[str]]
                         ) -> Dict[str, Union[str, SemanticType]]:
            col_type: Dict[str, Union[str, SemanticType]] = dict()
            for t, cols in types_to_cols.items():
                for c in cols:
                    col_type[c] = t
            return col_type

        column_type = dict()
        column_type.update(pivot_to_col(stats.schema_map))
        column_type.update(pivot_to_col(strong_semantic_detections))

        num_cols = 0
        for col_name, col_type in column_type.items():
            num_cols += self._get_num_values_in_col(col_name, col_type, stats, sparse_or_dense)

        if sparse_or_dense == SparseOrDense.SPARSE:
            num_cols *= self._SPARSE_MATRIX_INDEX_OVERHEAD

        num_rows_to_sample_from = stats.num_rows - stats.num_invalid_rows
        # num_cols * num_rows * size_of_each_elem * pct_of_data_in_training * scale_up_from_training
        mem_required = (
            num_cols * num_rows_to_sample_from * self._BYTES_IN_NUMPY_ARRAY_ELEMENT *
            self._TEST_TRAIN_SPLIT_FACTOR * self._MAX_TRAINING_SCALE_UP_FACTOR / BYTES_IN_MiB)

        _logger.debug(f"""[sparse_or_dense({sparse_or_dense})]
            mem_required_for_training ({mem_required}) =
            num_cols ({num_cols}) *
            num_rows_to_sample_from ({num_rows_to_sample_from}) *
            bytes_in_float64 (8) *
            _TEST_TRAIN_SPLIT_FACTOR ({self._TEST_TRAIN_SPLIT_FACTOR}) *
            _MAX_TRAINING_SCALE_UP_FACTOR ({self._MAX_TRAINING_SCALE_UP_FACTOR}) /
            BYTES_IN_MiB ({BYTES_IN_MiB})""")

        return mem_required

    def get_sampling_fraction(
            self, stats: PreSamplingStats,
            strong_semantic_detections: Dict[SemanticType, List[str]]) -> Optional[float]:
        """
        Analyze the dataset and determine the sampling fraction that will allow the dataset to
        fit into memory when converted to pandas and after feature engineering during training

        :param stats: Dataset stats for the pyspark dataset
        :param strong_semantic_detections: Dictionary with strongly detected semantic types and their column names
        :return: (sparse_or_dense, fraction)
          sparse_or_dense: whether to encode the data with sparse or dense matrix
          fraction: the fraction to be used to sample the dataset if it's < 1 else None
        """
        mem_req_data_load_mb = self._get_data_load_mem_req(stats=stats)
        mem_req_training_mb_dense = self._get_training_mem_req(
            stats=stats,
            strong_semantic_detections=strong_semantic_detections,
            sparse_or_dense=SparseOrDense.DENSE)
        mem_req_training_mb_sparse = self._get_training_mem_req(
            stats=stats,
            strong_semantic_detections=strong_semantic_detections,
            sparse_or_dense=SparseOrDense.SPARSE)
        if InternalConfs.ENABLE_SPARSE_MATRIX and (
                mem_req_training_mb_dense >
                mem_req_training_mb_sparse * self._SPARSE_COMPUTATION_OVERHEAD):
            sparse_or_dense = SparseOrDense.SPARSE
            mem_req_training_mb = mem_req_training_mb_sparse
        else:
            sparse_or_dense = SparseOrDense.DENSE
            mem_req_training_mb = mem_req_training_mb_dense
        _logger.debug(f"""
            mem_req_data_load_mb = {mem_req_data_load_mb}
            mem_req_training_mb_dense = {mem_req_training_mb_dense}
            mem_req_training_mb_sparse = {mem_req_training_mb_sparse}
            mem_req_training_mb = {mem_req_training_mb}""")

        max_memory_req_mb = max(mem_req_data_load_mb, mem_req_training_mb)
        fraction = self._memory_mb_per_trial / max_memory_req_mb
        _logger.debug(f"""fraction ({fraction}) =
            self._memory_per_trial_mb ({self._memory_mb_per_trial}) /
            max_memory_req_mb ({max_memory_req_mb})""")
        return SizeEstimatorResult(
            sparse_or_dense,
            mem_req_data_load_mb,
            mem_req_training_mb_dense,
            mem_req_training_mb_sparse,
            sample_fraction=fraction if fraction < 1.0 else None)
