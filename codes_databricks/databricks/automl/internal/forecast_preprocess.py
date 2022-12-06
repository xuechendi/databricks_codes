import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Type
from typing import Optional

from pyspark.sql.types import DataType, StructType

from databricks.automl.internal.alerts import TimeSeriesIdentitiesTooShortAlert, NotEnoughHistoricalDataAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.base_preprocess import BaseDataPreprocessor
from databricks.automl.internal.confs import ForecastConf
from databricks.automl.internal.stats import IntermediateStats
from databricks.automl.shared.errors import InvalidArgumentError, UnsupportedDataError

_logger = logging.getLogger(__name__)


@dataclass
class ForecastDataPreprocessResults:
    """
    Class for the data preprocess results
    """
    invalid_identities: List[str]
    target_col_type: Type[DataType]
    time_col_type: Type[DataType]
    num_folds: int


class ForecastDataPreprocessor(BaseDataPreprocessor):
    """
    Validate the dataset and extract information from it
    """
    MIN_TIME_SERIES_LENGTH = 5

    def __init__(self, intermediate_stats: IntermediateStats, dataset_schema: StructType,
                 target_col: str, time_col: str, identity_col: Optional[List[str]], horizon: int,
                 supported_target_types: List[DataType], supported_time_types: List[DataType],
                 alert_manager: AlertManager, confs: ForecastConf):

        super().__init__(
            intermediate_stats=intermediate_stats,
            dataset_schema=dataset_schema,
            alert_manager=alert_manager,
            target_col=target_col,
        )

        self._validate_identity_cols(identity_col)
        self._time_col_type = self._validate_col_type(time_col, "time_col", supported_time_types)
        self._target_col_type = self._validate_col_type(target_col, "target_col",
                                                        supported_target_types)

        self._num_folds = confs.num_folds
        if identity_col:
            self._selected_cols = [time_col, target_col] + identity_col
        else:
            self._selected_cols = [time_col, target_col]

    @property
    def selected_cols(self) -> Optional[List[str]]:
        return self._selected_cols

    @property
    def target_col_type(self) -> Type[DataType]:
        return self._target_col_type

    @property
    def time_col_type(self) -> Type[DataType]:
        return self._time_col_type

    def _validate_identity_cols(self, identity_col: Optional[List[str]]):
        if identity_col:
            for col in identity_col:
                if col not in self._dataset_schema.fieldNames():
                    raise InvalidArgumentError(f"identity_col:{col} does not exist.")

    def process_data_exploration_result(
            self, data_exp_result: Dict[str, Any]) -> ForecastDataPreprocessResults:
        time_series_count = data_exp_result["count"]

        valid_time_series = [
            k for k, v in time_series_count.items() if v >= self.MIN_TIME_SERIES_LENGTH
        ]
        invalid_identities = [
            k for k, v in time_series_count.items() if v < self.MIN_TIME_SERIES_LENGTH
        ]
        if not valid_time_series:
            self._alert_manager.record(NotEnoughHistoricalDataAlert())
            raise UnsupportedDataError(f"Not enough time series data for training and validation. "
                                       f"Please provide longer time series data.")
        elif len(invalid_identities) > 0:
            self._alert_manager.record(TimeSeriesIdentitiesTooShortAlert(invalid_identities))
            _logger.warning(
                f"Time series with the following identities are too short: {invalid_identities}. "
                f"Models won't be trained for these identities. "
                f"Please provide longer time series data.")

        preprocess_result = ForecastDataPreprocessResults(
            invalid_identities=invalid_identities,
            target_col_type=self._target_col_type,
            time_col_type=self._time_col_type,
            num_folds=self._num_folds)
        return preprocess_result
