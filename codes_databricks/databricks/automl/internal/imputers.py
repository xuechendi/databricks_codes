from abc import ABC, abstractmethod
from typing import Any, Union

import pandas as pd
from databricks.automl_runtime.sklearn import DatetimeImputer
from pyspark.sql.types import DataType
from sklearn.impute import SimpleImputer

from databricks.automl.shared.errors import InvalidArgumentError
from databricks.automl.internal.common.const import AutoMLDataType


class Imputer(ABC):
    imputable_types = [
        AutoMLDataType.BOOLEAN, AutoMLDataType.DATETIME, AutoMLDataType.NUMERIC,
        AutoMLDataType.STRING, AutoMLDataType.TEXT
    ]

    def __init__(self, type_: AutoMLDataType):
        self._type = type_

    @staticmethod
    def create_imputer(strategy: str, col: str, spark_type: DataType, fill_value: Any = None):
        valid_imputers = {
            "mean": ImputeMean,
            "median": ImputeMedian,
            "most_frequent": ImputeMostFrequent,
            "constant": ImputeConstant,
        }

        if strategy not in valid_imputers:
            raise InvalidArgumentError(
                f"Invalid imputation strategy {strategy} for column {col}. Imputation strategy must be one of {list(valid_imputers.keys())}."
            )
        type_ = AutoMLDataType.from_spark_type(spark_type)
        if type_ not in Imputer.imputable_types:
            raise InvalidArgumentError(
                f"Imputing {spark_type} column is not supported, for column {col}.")

        imputer_class = valid_imputers[strategy]
        if imputer_class == ImputeConstant:
            return ImputeConstant(spark_type, fill_value, col)
        elif imputer_class in (ImputeMean, ImputeMedian):
            if type_ in (AutoMLDataType.STRING, AutoMLDataType.BOOLEAN):
                raise InvalidArgumentError(
                    f"Invalid imputation strategy {strategy} for column {col} with type {spark_type}."
                )
            else:
                return imputer_class(type_)
        else:
            return imputer_class(type_)

    @property
    def type(self) -> AutoMLDataType:
        return self._type

    @abstractmethod
    def get_name(self) -> str:
        """
        :return: name of the imputer used in training notebook Pipelines
        """
        pass

    @abstractmethod
    def get_sklearn_imputer(self) -> Union[SimpleImputer, DatetimeImputer]:
        pass

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self._type == other._type

    def __hash__(self):
        return hash((self.__class__.__name__, self._type))


class ImputeMean(Imputer):
    def get_name(self) -> str:
        return "impute_mean"

    def get_sklearn_imputer(self) -> Union[SimpleImputer, DatetimeImputer]:
        if self._type == AutoMLDataType.DATETIME:
            return DatetimeImputer(strategy="mean")
        else:
            return SimpleImputer(strategy="mean")


class ImputeMedian(Imputer):
    def get_name(self) -> str:
        return "impute_median"

    def get_sklearn_imputer(self) -> Union[SimpleImputer, DatetimeImputer]:
        if self._type == AutoMLDataType.DATETIME:
            return DatetimeImputer(strategy="median")
        else:
            return SimpleImputer(strategy="median")


class ImputeMostFrequent(Imputer):
    def get_name(self) -> str:
        return "impute_most_frequent"

    def get_sklearn_imputer(self) -> Union[SimpleImputer, DatetimeImputer]:
        if self._type == AutoMLDataType.DATETIME:
            return DatetimeImputer(strategy="most_frequent")
        elif self._type in (AutoMLDataType.STRING, AutoMLDataType.BOOLEAN):
            return SimpleImputer(missing_values=None, strategy="most_frequent")
        else:
            return SimpleImputer(strategy="most_frequent")


class ImputeConstant(Imputer):
    def __init__(self, spark_type: DataType, fill_value: Any, col: str = ""):
        if fill_value is None:
            raise InvalidArgumentError(
                f"Must provide 'fill_value' for 'constant' imputation strategy, for column {col}.")

        # Input validation: make sure the `fill_value` is consistent with the `spark_type`.
        type_ = AutoMLDataType.from_spark_type(spark_type)
        if type_ == AutoMLDataType.NUMERIC:
            try:
                fill_value = float(fill_value)
            except Exception as e:
                raise InvalidArgumentError(
                    f"Column {col} has type {spark_type}, but unable to convert fill_value {fill_value} into a number."
                )
        elif type_ == AutoMLDataType.STRING:
            try:
                fill_value = str(fill_value)
            except Exception as e:
                # must use {repr(fill_value)}, because {fill_value} will call __str__ and raise an exception
                raise InvalidArgumentError(
                    f"Column {col} has type {spark_type}, but unable to convert fill_value {repr(fill_value)} into a string."
                )
        elif type_ == AutoMLDataType.BOOLEAN:
            if not isinstance(fill_value, bool):
                # if fill_value is "true" / "True" / "false" / "False" the convert it
                if isinstance(fill_value, str) and fill_value.lower() in {"true", "false"}:
                    fill_value = (fill_value.lower() == "true")
                else:
                    raise InvalidArgumentError(
                        f"Column {col} has type {spark_type}, but unable to convert fill_value {fill_value} into a boolean."
                    )
        elif type_ == AutoMLDataType.DATETIME:
            try:
                fill_value = pd.to_datetime(fill_value)
            except Exception as e:
                raise InvalidArgumentError(
                    f"Column {col} has type {spark_type}, but unable to convert fill_value {fill_value} into a "
                    f"datetime object.")
        else:
            raise InvalidArgumentError(
                f"Imputation with constant not supported for column {col} with type {spark_type}.")

        super().__init__(type_)
        self.fill_value = fill_value

    def get_name(self) -> str:
        return f"impute_{self._type.value.lower()}_{self.fill_value}"

    def get_sklearn_imputer(self) -> Union[SimpleImputer, DatetimeImputer]:
        if self._type == AutoMLDataType.DATETIME:
            return DatetimeImputer(strategy="constant", fill_value=self.fill_value)
        if self._type in (AutoMLDataType.STRING, AutoMLDataType.BOOLEAN):
            return SimpleImputer(
                missing_values=None,
                strategy="constant",
                fill_value=self.fill_value,
            )
        else:
            return SimpleImputer(strategy="constant", fill_value=self.fill_value)

    def __eq__(self, other):
        return super().__eq__(
            other) and self.fill_value == other.fill_value and self._type == other._type

    def __hash__(self):
        return hash((self.__class__.__name__, self._type, self.fill_value))
