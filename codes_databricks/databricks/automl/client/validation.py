import logging
from typing import Union, List, Optional, Dict, Tuple
from collections import namedtuple
import uuid

import pandas as pd
import pyspark.pandas as ps
import pyspark.sql
from pyspark.sql.dataframe import StructType
from pyspark.sql.session import SparkSession

from databricks.automl.shared import utils as shared_utils
from databricks.automl.client.protos.common_pb2 import Imputer
from databricks.automl.client.protos.forecasting_pb2 import ForecastingParams
from databricks.automl.shared.const import GLOBAL_TEMP_DATABASE, TimeSeriesFrequency
from databricks.automl.shared.errors import InvalidArgumentError

_logger = logging.getLogger(__name__)

InputColumnParam = namedtuple("InputColumnParam", ["name", "input_cols", "required"])


class InputValidator:
    @staticmethod
    def get_dataframe_and_name(
            dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame, str]
    ) -> Tuple[pyspark.sql.DataFrame, str]:
        """
        Returns a pyspark.sql.dataframe and its name
        """
        spark = SparkSession.builder.getOrCreate()
        if isinstance(dataset, str):
            dataframe = spark.table(dataset)
            return dataframe, dataset
        else:
            dataframe = shared_utils.convert_to_spark_dataframe(dataset)
            name = f"automl_{str(uuid.uuid4()).replace('-', '_')}"  # table name cannot have dashes
            dataframe.createOrReplaceGlobalTempView(name)
            return dataframe, f"{GLOBAL_TEMP_DATABASE}.{name}"

    @staticmethod
    def validate_cols_exists(schema: StructType, params: List[InputColumnParam]) -> None:
        for param in params:
            if not param.required and not param.input_cols:
                continue

            if param.required and not param.input_cols:
                raise InvalidArgumentError(f"Param: {param.name} is required but not passed")

            if isinstance(param.input_cols, str):
                cols_to_check = [param.input_cols]
            else:
                cols_to_check = param.input_cols

            for col in cols_to_check:
                if col not in schema.fieldNames():
                    raise InvalidArgumentError(
                        f"Dataset schema does not contain column with name '{col}'. "
                        f"Please pass a valid column name for param: {param.name}")

    @staticmethod
    def parse_frequency(frequency: str) -> ForecastingParams.Frequency:
        try:
            frequency_enum = TimeSeriesFrequency[frequency]
        except KeyError:
            raise InvalidArgumentError(
                f"Unknown value frequency={frequency}. Please provide a valid value.")
        frequency_proto_enum = ForecastingParams.Frequency.Value(frequency_enum.proto_enum_value)
        return frequency_proto_enum

    @staticmethod
    def parse_imputers(imputers: Optional[Dict[str, Union[str, Dict]]]) -> List[Imputer]:
        if imputers is None:
            return []

        imputer_protos = []
        for col_name, imputer in imputers.items():
            if isinstance(imputer, str):
                strategy = Imputer.Strategy.Value(imputer.upper())
                fill_value = None
            else:
                strategy = imputer.get("strategy", None)
                fill_value = imputer.get("fill_value", None)

                # convert all non-null fill values to str
                if fill_value is not None:
                    fill_value = str(fill_value)

                if strategy is not None:
                    strategy = Imputer.Strategy.Value(strategy.upper())

            imputer_proto = Imputer(col_name=col_name, strategy=strategy, fill_value=fill_value)
            imputer_protos.append(imputer_proto)
        return imputer_protos

    @staticmethod
    def warn_if_max_trials(max_trials: Optional[int]) -> None:
        if max_trials:
            _logger.warning(
                "Parameter max_trials is deprecated and has no effect. This parameter will be removed in a future "
                "Databricks Runtime release. Use timeout_minutes to control the duration of an AutoML experiment. "
                "AutoML will automatically stop tuning models if the validation metric no longer improves."
            )

    @staticmethod
    def consolidate_exclude_cols_params(exclude_columns: Optional[List[str]] = None,
                                        exclude_cols: Optional[List[str]] = None
                                        ) -> Optional[List[str]]:
        if exclude_columns:
            _logger.warning(
                "Parameter exclude_columns is deprecated and will be removed in a future Databricks Runtime release. "
                "Please use exclude_cols instead.")
            if exclude_cols:
                _logger.warning("Both exclude_columns and exclude_cols are specified. "
                                "The value of param exclude_columns will be ignored.")
                return exclude_cols
            return exclude_columns
        return exclude_cols
