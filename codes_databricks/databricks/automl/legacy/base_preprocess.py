import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Type, Union, Mapping
from typing import Optional

from pyspark.sql.types import ArrayType, DataType, ByteType, ShortType, IntegerType, LongType, FloatType, \
    DoubleType, BooleanType, StringType, TimestampType, DateType, StructType

from databricks.automl.legacy.alerts import DatasetEmptyAlert, NoFeatureColumnsAlert, UnsupportedTargetTypeAlert, \
    UnsupportedTimeTypeAlert, AllRowsInvalidAlert
from databricks.automl.legacy.alerts.alert_manager import AlertManager
from databricks.automl.legacy.const import DatasetFormat
from databricks.automl.legacy.errors import InvalidArgumentError, UnsupportedDataError, UnsupportedColumnError
from databricks.automl.legacy.stats import PreSamplingStats

_logger = logging.getLogger(__name__)


class BaseDataPreprocessor(ABC):
    """
    Validate the dataset and extract information from it
    """
    INTEGER_TYPES = (
        ByteType,
        ShortType,
        IntegerType,
        LongType,
    )
    FLOATING_POINT_TYPES = (
        FloatType,
        DoubleType,
    )
    NUMERIC_TYPES = INTEGER_TYPES + FLOATING_POINT_TYPES
    TIME_TYPES = (TimestampType, DateType)
    STRING_TYPE = (StringType, )
    ALL_TYPES = NUMERIC_TYPES + TIME_TYPES + STRING_TYPE + (ArrayType, BooleanType)
    SPARK_TYPE_TO_PANDAS_TYPE = {"NumericType": "float", "TimestampType": "datetime"}

    UNSUPPORTED_TYPE = "unsupported"
    NUMERICAL_TYPE = "numeric"

    def __init__(self, pre_sampling_stats: PreSamplingStats, dataset_schema: StructType,
                 dataset_format: DatasetFormat, alert_manager: AlertManager, target_col: str):
        self._pre_sampling_stats = pre_sampling_stats
        self._dataset_schema = dataset_schema
        self._dataset_format = dataset_format
        self._alert_manager = alert_manager
        self._target_col = target_col

        self._validate_dataset_has_rows()

    @property
    @abstractmethod
    def selected_cols(self) -> Optional[List[str]]:
        pass

    @property
    @abstractmethod
    def target_col_type(self) -> Type[DataType]:
        pass

    def _validate_col_type(self, col_name: str, arg_name: str,
                           supported_data_types: List[DataType]) -> Type[DataType]:
        """
        Validates that the given column name exists in the schema and is of a supported type
        and returns the type of the column

        :param col_name: Name of the column to validate
        :param arg_name: Name of the argument that represents this column name
        :param supported_data_types: List of supported types for this column
        :return: Actual type of the column
        """
        if col_name not in self._dataset_schema.fieldNames():
            raise InvalidArgumentError(f"{arg_name}={col_name} does not exist.")
        col_type = self._dataset_schema[col_name].dataType
        if not isinstance(col_type, supported_data_types):
            if arg_name == "time_col":
                self._alert_manager.record(UnsupportedTimeTypeAlert(col_name, str(col_type)))
            elif arg_name == "target_col":
                self._alert_manager.record(UnsupportedTargetTypeAlert(col_name, str(col_type)))
            raise UnsupportedColumnError(
                f"Column \"{col_name}\" of type {col_type} is not supported. "
                f"Supported types: {tuple(t() for t in supported_data_types)}")
        col_type = type(col_type)
        return col_type

    @dataclass
    class SchemaInfo:
        # Mapping from types to column names
        feature_schema: Mapping[Union[str, Type[DataType]], List[str]]
        # Metadata for each column name
        feature_metadata: Mapping[str, Dict]
        # Supported column names
        supported_cols: List[str]

    def _get_schema_info(self) -> SchemaInfo:
        """
        Unpacks schema information in the input dataset.

        :param target_col: name of target column
        :return: A FeatureSchema data class
        """
        feature_schema = defaultdict(list)
        supported_cols = []
        fields = [field for field in self._dataset_schema.fields if field.name != self._target_col]
        feature_metadata = dict()
        for field in fields:
            if isinstance(field.dataType, self.ALL_TYPES):
                # condense feature schema into format {dtype: [column_names]}
                # collapse numerical types into one
                supported_cols.append(field.name)
                if isinstance(field.dataType, self.NUMERIC_TYPES):
                    feature_schema[self.NUMERICAL_TYPE].append(field.name)
                else:
                    feature_schema[type(field.dataType)].append(field.name)
            else:
                feature_schema[self.UNSUPPORTED_TYPE].append(field.name)
            feature_metadata[field.name] = field.metadata

        # calculate supported cols that have all nulls
        num_rows = self._pre_sampling_stats.num_rows
        empty_supported_cols = [
            col for col in supported_cols
            if self._pre_sampling_stats.columns[col].num_nulls == num_rows
        ]
        if len(empty_supported_cols) > 0:
            # only log to console, these will be detected as constant columns later on and dropped in training
            _logger.info(
                f"Following columns are found to have all nulls and will be dropped: {empty_supported_cols}"
            )

        supported_non_empty_cols = set(supported_cols) - set(empty_supported_cols)

        if len(supported_non_empty_cols) == 0:
            self._alert_manager.record(NoFeatureColumnsAlert())
            raise UnsupportedDataError("No supported column types found in dataset")
        supported_cols.append(self._target_col)
        return self.SchemaInfo(
            feature_schema=feature_schema,
            supported_cols=supported_cols,
            feature_metadata=feature_metadata)

    def _validate_dataset_has_rows(self) -> None:
        if self._pre_sampling_stats.num_rows == 0:
            self._alert_manager.record(DatasetEmptyAlert())
            raise UnsupportedDataError("The input dataset is empty. Please pass in a valid dataset.")

        if self._pre_sampling_stats.num_rows == self._pre_sampling_stats.num_invalid_rows:
            self._alert_manager.record(AllRowsInvalidAlert(self._target_col))
            raise UnsupportedDataError(
                f"Every value in the selected target_col {self._target_col} is either null "
                "or does not have enough rows (5) per target class. Please pass in a valid dataset.")
