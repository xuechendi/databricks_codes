from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyspark.pandas as ps
import pyspark.sql

from databricks.automl.legacy.classifier import Classifier
from databricks.automl.legacy.confs import UnivariateForecastConf
from databricks.automl.legacy.const import ContextType
from databricks.automl.legacy.forecast import Forecast
from databricks.automl.legacy.problem_type import ClassificationTargetTypes, Metric
from databricks.automl.legacy.regressor import Regressor
from databricks.automl.legacy.result import AutoMLSummary
from databricks.automl.legacy.supervised_learner import SupervisedLearner


# For user-facing APIs in this file, put required parameters before optional parameters,
# and then use alphabetical order.
def classify(
        dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame],
        *,
        target_col: str,
        data_dir: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        exclude_frameworks: Optional[List[str]] = None,
        experiment_dir: Optional[str] = None,
        imputers: Optional[Dict[str, Union[str, Dict]]] = None,
        # `max_trials` is deprecated and only checked for deprecation warning
        max_trials: Optional[int] = None,
        primary_metric: str = "f1",
        pos_label: Optional[ClassificationTargetTypes] = None,
        time_col: Optional[str] = None,
        timeout_minutes: Optional[int] = SupervisedLearner.DEFAULT_TIMEOUT_MINUTES
) -> AutoMLSummary:
    """
    Automatically generates trial notebooks and trains classification models.

    :param dataset:         input Spark or pandas DataFrame that contains training features and targets.
    :param target_col:      column name of the target labels.
    :param data_dir:        Optional DBFS path that is visible to both driver and worker nodes used to store
                            intermediate data
    :param exclude_cols:    columns that will be ignored by AutoML
    :param exclude_columns: columns that will be ignored by AutoML
                            .. warning:: Parameter exclude_cols is deprecated and will be removed in a future
                            Databricks Runtime release. Please use exclude_cols instead.
    :param exclude_frameworks: Frameworks that will be ignored by AutoML. The supported frameworks are "sklearn",
                               "xgboost", and "lightgbm".
    :param experiment_dir:  Optional workspace path where the generated notebooks and experiments will be saved. If not
                            provided, AutoML will default to saving these under /Users/<username>/databricks_automl/
    :param imputers:        Dictionary where each key is a column name, and each value is a string or dictionary
                            describing the imputation strategy. If specified as a string, the imputation strategy must
                            be one of "mean", "median", or "most_frequent". To impute with a known value, specify the
                            imputation strategy as a dictionary {"strategy": "constant", value: <desired value>}.
                            String options can also be specified as dictionaries, for example {"strategy": "mean"}. If
                            no imputation strategy is provided for a column, a default strategy will be selected by
                            AutoML.
                            Example: imputers={ "col1": "median", "col2": {"strategy": "constant", "fill_value": 25} }
    :param max_trials:      .. warning:: Parameter max_trials is deprecated and has no effect. The parameter will be
                            removed in a future Databricks Runtime release. Choosing timeout_minutes to control the
                            AutoML runs. AutoML stops training early and tuning models if the validation metric is no
                            longer improving.
    :param primary_metric:  primary metric to select the best model. Each trial will compute several metrics, but this
                            one determines which model is selected from all the trials. One of "f1" (default),
                            "log_loss", "accuracy", "precision", "roc_auc".
    :param pos_label:       The positive class, useful for calculating metrics such as precision/recall. Should only be
                            specified in binary classification use case.
    :param time_col:        Optional column name of a time column. If provided, AutoML will try to split train/val/test
                            sets by time. Accepted column types are date/time, string and integer. If column type is
                            string AutoML will try to convert it to datetime by semantic detection, and the AutoML run
                            will fail if the conversion fails.
    :param timeout_minutes: The maximum time to wait for the AutoML trials to complete. timeout_minutes=None will run
                            the trials without any timeout restrictions. The default value is 120 minutes.

    :return: Structured summary object with info about trials.
    """
    return Classifier(context_type=ContextType.DATABRICKS).fit(
        dataset=dataset,
        target_col=target_col,
        data_dir=data_dir,
        exclude_cols=exclude_cols,
        exclude_columns=exclude_columns,
        exclude_frameworks=exclude_frameworks,
        home_dir=experiment_dir,
        imputers=imputers,
        max_trials=max_trials,
        metric=primary_metric,
        pos_label=pos_label,
        time_col=time_col,
        timeout_minutes=timeout_minutes)


def regress(
        dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame],
        *,
        target_col: str,
        data_dir: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        exclude_frameworks: Optional[List[str]] = None,
        experiment_dir: Optional[str] = None,
        imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
        # `max_trials` is deprecated and only checked for deprecation warning
        max_trials: Optional[int] = None,
        primary_metric: str = "r2",
        time_col: Optional[str] = None,
        timeout_minutes: Optional[int] = SupervisedLearner.DEFAULT_TIMEOUT_MINUTES
) -> AutoMLSummary:
    """
    Automatically generates trial notebooks and trains regression models.

    :param dataset:         input Spark or pandas DataFrame that contains training features and targets.
    :param target_col:      column name of the target labels.
    :param data_dir:        Optional DBFS path that is visible to both driver and worker nodes used to store
                            intermediate data.
    :param exclude_cols:    columns that will be ignored by AutoML
    :param exclude_columns: columns that will be ignored by AutoML
                            .. warning:: Parameter exclude_cols is deprecated and will be removed in a future
                            Databricks Runtime release. Please use exclude_cols instead.
    :param exclude_frameworks: Frameworks that will be ignored by AutoML. The supported frameworks are "sklearn",
                            "xgboost", and "lightgbm".
    :param experiment_dir:  Optional workspace path where the generated notebooks and experiments will be saved. If not
                            provided, AutoML will default to saving these under /Users/<username>/databricks_automl/
    :param imputers:        Dictionary where each key is a column name, and each value is a string or dictionary
                            describing the imputation strategy. If specified as a string, the imputation strategy must
                            be one of "mean", "median", or "most_frequent". To impute with a known value, specify the
                            imputation strategy as a dictionary {"strategy": "constant", value: <desired value>}.
                            String options can also be specified as dictionaries, for example {"strategy": "mean"}. If
                            no imputation strategy is provided for a column, a default strategy will be selected by
                            AutoML.
                            Example: imputers={ "col1": "median", "col2": {"strategy": "constant", "fill_value": 25} }
    :param max_trials:      .. warning:: Parameter max_trials is deprecated and has no effect. The parameter will be
                            removed in a future Databricks Runtime release. Choosing timeout_minutes to control the
                            AutoML runs. AutoML stops training early and tuning models if the validation metric is no
                            longer improving.
    :param primary_metric:  primary metric to select the best model. Each trial will compute several metrics, but this
                            one determines which model is selected from all the trials. One of "r2" (default,
                            R squared), "mse" (mean squared error), "rmse" (root mean squared error),
                            "mae" (mean absolute error).
    :param time_col:        Optional column name of a time column. If provided, AutoML will try to split train/val/test
                            sets by time. Accepted column types are date/time, string and integer. If column type is
                            string AutoML will try to convert it to datetime by semantic detection, and the AutoML run
                            will fail if the conversion fails.
    :param timeout_minutes: The maximum time to wait for the AutoML trials to complete. timeout_minutes=None will run
                            the trials without any timeout restrictions. The default value is 120 minutes.

    :return: Structured summary object with info about trials.
    """
    return Regressor(context_type=ContextType.DATABRICKS).fit(
        dataset=dataset,
        target_col=target_col,
        data_dir=data_dir,
        exclude_cols=exclude_cols,
        exclude_columns=exclude_columns,
        exclude_frameworks=exclude_frameworks,
        home_dir=experiment_dir,
        imputers=imputers,
        max_trials=max_trials,
        metric=primary_metric,
        time_col=time_col,
        timeout_minutes=timeout_minutes)


def forecast(dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame],
             *,
             target_col: str,
             time_col: str,
             data_dir: Optional[str] = None,
             exclude_frameworks: Optional[List[str]] = None,
             experiment_dir: Optional[str] = None,
             frequency: str = "D",
             horizon: int = 1,
             identity_col: Optional[Union[str, list]] = None,
             max_trials: Optional[int] = UnivariateForecastConf().max_trials,
             primary_metric: str = "smape",
             timeout_minutes: Optional[int] = Forecast.DEFAULT_TIMEOUT_MINUTES,
             output_database: Optional[str] = None) -> AutoMLSummary:
    """
    Automatically generates trial notebooks and trains time series forecasting model.

    :param dataset: input Spark or pandas DataFrame that contains training features and targets.
    :param target_col: column name of the target labels.
    :param time_col: column name of the time column for forecasting.
    :param data_dir: Optional DBFS path that is visible to both driver and worker nodes used to store
                    intermediate data.
    :param exclude_frameworks: Frameworks that will be ignored by AutoML. The supported frameworks are "prophet"
                               and "arima".
    :param experiment_dir:  Optional workspace path where the generated notebooks and experiments will be saved. If not
                            provided, AutoML will default to saving these under /Users/<username>/databricks_automl/
    :param frequency: frequency of the time series. This represents the period with which events are expected to occur.
                        Possible values:
                        * "W" (weeks)
                        * "D" / "days" / "day"
                        * "hours" / "hour" / "hr" / "h"
                        * "m" / "minute" / "min" / "minutes" / "T"
                        * "S" / "seconds" / "sec" / "second"
    :param horizon: the length of time into the future for which forecasts are to be prepared. The horizon is in units
                        of the time series frequency.
    :param identity_col: Optional list of column names of the identity columns for multi-series forecasting.
    :param max_trials:      The maximum number of trials to run.
    :param primary_metric: primary metric to select the best model. Each trial will compute several metrics, but this
                           one determines which model is selected from all the trials. One of "smape"(default) "mse",
                           "rmse", "mae", "mdape".
    :param timeout_minutes: The maximum time to wait for the AutoML trials to complete. timeout_minutes=None will run
                            the trials without any timeout restrictions. The default value is 120 minutes.
    :param output_database: Schema name to save the predicted data. AutoML will create a new table in the database
                            with the predicted data. If it is None, AutoML will not save any results.

    :return: Structured summary object with info about trials.
    """
    return Forecast(context_type=ContextType.DATABRICKS).fit(
        dataset=dataset,
        target_col=target_col,
        time_col=time_col,
        data_dir=data_dir,
        exclude_frameworks=exclude_frameworks,
        home_dir=experiment_dir,
        frequency=frequency,
        horizon=horizon,
        identity_col=identity_col,
        max_trials=max_trials,
        metric=primary_metric,
        timeout_minutes=timeout_minutes,
        output_database=output_database)
