import logging
from abc import ABC, abstractmethod
from typing import List, Type
from typing import Optional

from pyspark.sql.types import DataType, StructType

from databricks.automl.internal.alerts import UnsupportedTargetTypeAlert, UnsupportedTimeTypeAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.shared.errors import InvalidArgumentError
from databricks.automl.internal.errors import UnsupportedColumnError
from databricks.automl.internal.stats import IntermediateStats

_logger = logging.getLogger(__name__)


class BaseDataPreprocessor(ABC):
    """
    Validate the dataset and extract information from it
    """

    def __init__(self, intermediate_stats: IntermediateStats, dataset_schema: StructType,
                 alert_manager: AlertManager, target_col: str):
        self._intermediate_stats = intermediate_stats
        self._dataset_schema = dataset_schema
        self._alert_manager = alert_manager
        self._target_col = target_col

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
