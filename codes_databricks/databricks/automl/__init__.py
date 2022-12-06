"""
This file contains user-facing APIs.
ContextType is not a user-facing API, but must be included here because it is used in AutoMLDriverNotebook.py
"""
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyspark.pandas as ps
import pyspark.sql

from databricks.automl import internal, legacy
from databricks.automl.version import VERSION as __version__

from databricks.automl.client.manager import AutomlLifeCycleManager
from databricks.automl.internal.classifier import Classifier
from databricks.automl.internal.errors import UnsupportedParameterError
from databricks.automl.internal.confs import UnivariateForecastConf
from databricks.automl.internal.forecast import Forecast
from databricks.automl.internal.common.const import ContextType
from databricks.automl.internal.regressor import Regressor
from databricks.automl.internal.supervised_learner import SupervisedLearner
from databricks.automl.internal.utils.logging_utils import _configure_automl_loggers
from databricks.automl.shared import utils as shared_utils
from databricks.automl.shared.const import ClassificationTargetTypes
from databricks.automl.shared.result import AutoMLSummary

# Required because driver notebook imported by the webapp uses
# import databricks.automl; databricks.automl.classifier.Classify(..)
# https://src.dev.databricks.com/databricks/universe@master/-/blob/webapp/web/js/mlflow/autoML/AutoMLDriverNotebook.py?L116
# Don't import forecast as it will conflict with the imported forecast function above
import databricks.automl.classifier
import databricks.automl.regressor

_configure_automl_loggers(root_module_name=__name__)


# For user-facing APIs in this file, put required parameters before optional parameters,
# and then use alphabetical order.
def classify(
        dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame, str],
        *,
        target_col: str,
        data_dir: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        exclude_frameworks: Optional[List[str]] = None,
        experiment_dir: Optional[str] = None,
        feature_store_lookups: Optional[List[Dict]] = None,
        imputers: Optional[Dict[str, Union[str, Dict]]] = None,
        # `max_trials` is deprecated and only checked for deprecation warning
        max_trials: Optional[int] = None,
        primary_metric: str = "f1",
        pos_label: Optional[ClassificationTargetTypes] = None,
        time_col: Optional[str] = None,
        timeout_minutes: Optional[int] = SupervisedLearner.DEFAULT_TIMEOUT_MINUTES
) -> Optional[AutoMLSummary]:
    """
    Automatically generates trial notebooks and trains classification models.

    :param dataset:         table name or spark/pandas/pyspark.pandas dataframe that contains features and targets
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
    :param feature_store_lookups: Optional list of dictionaries that represent features from Feature Store for data
                                  augmentation. Valid keys in each dictionary are
                                  * table_name (str): Required. Name of the feature table
                                  * lookup_key (list or str): Required. Column name(s) to be used as key when joining
                                    the feature table with the data passed in the `dataset` param. The order of the
                                    column names must match the order of the primary keys of the feature table.
                                  * timestamp_lookup_key (str): Only required if the specified table is a time series
                                    feature table. The column name to be used when performing point-in-time lookup
                                    on the feature table with the data passed in the `dataset` param.
                                  This feature is available only on E2 version of the Databricks platform.
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

    :return: Structured summary object with info about trials, or None if the summary could not be loaded.
    """
    if shared_utils.is_automl_service_enabled():
        if shared_utils.use_automl_client():
            return AutomlLifeCycleManager().execute_classification(
                dataset=dataset,
                target_col=target_col,
                data_dir=data_dir,
                exclude_cols=exclude_cols,
                exclude_columns=exclude_columns,
                exclude_frameworks=exclude_frameworks,
                experiment_dir=experiment_dir,
                feature_store_lookups=feature_store_lookups,
                imputers=imputers,
                max_trials=max_trials,
                primary_metric=primary_metric,
                pos_label=pos_label,
                time_col=time_col,
                timeout_minutes=timeout_minutes)
        else:
            return internal.classifier.Classifier(context_type=ContextType.DATABRICKS).fit(
                dataset=dataset,
                target_col=target_col,
                data_dir=data_dir,
                exclude_cols=exclude_cols,
                exclude_columns=exclude_columns,
                exclude_frameworks=exclude_frameworks,
                home_dir=experiment_dir,
                feature_store_lookups=feature_store_lookups,
                imputers=imputers,
                max_trials=max_trials,
                metric=primary_metric,
                pos_label=pos_label,
                time_col=time_col,
                timeout_minutes=timeout_minutes)
    else:
        if feature_store_lookups is not None:
            raise UnsupportedParameterError(
                "classify() got an unexpected keyword argument 'feature_store_lookups'")
        return legacy.classify(
            dataset=dataset,
            target_col=target_col,
            data_dir=data_dir,
            exclude_cols=exclude_cols,
            exclude_columns=exclude_columns,
            exclude_frameworks=exclude_frameworks,
            experiment_dir=experiment_dir,
            imputers=imputers,
            max_trials=max_trials,
            primary_metric=primary_metric,
            pos_label=pos_label,
            time_col=time_col,
            timeout_minutes=timeout_minutes)


def regress(
        dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame, str],
        *,
        target_col: str,
        data_dir: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        exclude_frameworks: Optional[List[str]] = None,
        experiment_dir: Optional[str] = None,
        feature_store_lookups: Optional[List[Dict]] = None,
        imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
        # `max_trials` is deprecated and only checked for deprecation warning
        max_trials: Optional[int] = None,
        primary_metric: str = "r2",
        time_col: Optional[str] = None,
        timeout_minutes: Optional[int] = SupervisedLearner.DEFAULT_TIMEOUT_MINUTES
) -> Optional[AutoMLSummary]:
    """
    Automatically generates trial notebooks and trains regression models.

    :param dataset:         table name or spark/pandas/pyspark.pandas dataframe that contains features and targets
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
    :param feature_store_lookups: Optional list of dictionaries that represent features from Feature Store for data
                                  augmentation. Valid keys in each dictionary are
                                  * table_name (str): Required. Name of the feature table
                                  * lookup_key (list or str): Required. Column name(s) to be used as key when joining
                                    the feature table with the data passed in the `dataset` param. The order of the
                                    column names must match the order of the primary keys of the feature table.
                                  * timestamp_lookup_key (str): Only required if the specified table is a time series
                                    feature table. The column name to be used when performing point-in-time lookup
                                    on the feature table with the data passed in the `dataset` param.
                                  This feature is available only on E2 version of the Databricks platform.
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

    :return: Structured summary object with info about trials, or None if the summary could not be loaded.
    """
    if shared_utils.is_automl_service_enabled():
        if shared_utils.use_automl_client():
            return AutomlLifeCycleManager().execute_regression(
                dataset=dataset,
                target_col=target_col,
                data_dir=data_dir,
                exclude_cols=exclude_cols,
                exclude_columns=exclude_columns,
                exclude_frameworks=exclude_frameworks,
                experiment_dir=experiment_dir,
                feature_store_lookups=feature_store_lookups,
                imputers=imputers,
                max_trials=max_trials,
                primary_metric=primary_metric,
                time_col=time_col,
                timeout_minutes=timeout_minutes)
        else:
            return internal.regressor.Regressor(context_type=ContextType.DATABRICKS).fit(
                dataset=dataset,
                target_col=target_col,
                data_dir=data_dir,
                exclude_cols=exclude_cols,
                exclude_columns=exclude_columns,
                exclude_frameworks=exclude_frameworks,
                home_dir=experiment_dir,
                feature_store_lookups=feature_store_lookups,
                imputers=imputers,
                max_trials=max_trials,
                metric=primary_metric,
                time_col=time_col,
                timeout_minutes=timeout_minutes)
    else:
        if feature_store_lookups is not None:
            raise UnsupportedParameterError(
                "regress() got an unexpected keyword argument 'feature_store_lookups'")
        return legacy.regress(
            dataset=dataset,
            target_col=target_col,
            data_dir=data_dir,
            exclude_cols=exclude_cols,
            exclude_columns=exclude_columns,
            exclude_frameworks=exclude_frameworks,
            experiment_dir=experiment_dir,
            imputers=imputers,
            max_trials=max_trials,
            primary_metric=primary_metric,
            time_col=time_col,
            timeout_minutes=timeout_minutes)


def forecast(
        dataset: Union[pyspark.sql.DataFrame, pd.DataFrame, ps.DataFrame, str],
        *,
        target_col: str,
        time_col: str,
        data_dir: Optional[str] = None,
        exclude_frameworks: Optional[List[str]] = None,
        experiment_dir: Optional[str] = None,
        frequency: str = "D",
        horizon: int = 1,
        identity_col: Optional[Union[str, list]] = None,
        # `max_trials` is deprecated and only checked for deprecation warning
        max_trials: Optional[int] = None,
        primary_metric: str = "smape",
        timeout_minutes: Optional[int] = Forecast.DEFAULT_TIMEOUT_MINUTES,
        output_database: Optional[str] = None) -> Optional[AutoMLSummary]:
    """
    Automatically generates trial notebooks and trains time series forecasting model.

    :param dataset: table name or spark/pandas/pyspark.pandas dataframe that contains features and targets
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
    :param max_trials:      .. warning:: Parameter max_trials is deprecated. The parameter will be removed in a future
                            Databricks Runtime release. Choosing timeout_minutes to control the AutoML runs.
    :param primary_metric: primary metric to select the best model. Each trial will compute several metrics, but this
                           one determines which model is selected from all the trials. One of "smape"(default) "mse",
                           "rmse", "mae", "mdape".
    :param timeout_minutes: The maximum time to wait for the AutoML trials to complete. timeout_minutes=None will run
                            the trials without any timeout restrictions. The default value is 120 minutes.
    :param output_database: Schema name to save the predicted data. AutoML will create a new table in the database
                            with the predicted data. If it is None, AutoML will not save any results.

    :return: Structured summary object with info about trials, or None if the summary could not be loaded.
    """
    if shared_utils.is_automl_service_enabled():
        if shared_utils.use_automl_client():
            return AutomlLifeCycleManager().execute_forecasting(
                dataset=dataset,
                target_col=target_col,
                time_col=time_col,
                data_dir=data_dir,
                exclude_frameworks=exclude_frameworks,
                experiment_dir=experiment_dir,
                frequency=frequency,
                horizon=horizon,
                identity_col=identity_col,
                max_trials=max_trials,
                primary_metric=primary_metric,
                timeout_minutes=timeout_minutes,
                output_database=output_database)
        else:
            return internal.forecast.Forecast(context_type=ContextType.DATABRICKS).fit(
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
    else:
        return legacy.forecast(
            dataset=dataset,
            target_col=target_col,
            time_col=time_col,
            data_dir=data_dir,
            exclude_frameworks=exclude_frameworks,
            experiment_dir=experiment_dir,
            frequency=frequency,
            horizon=horizon,
            identity_col=identity_col,
            max_trials=max_trials,
            primary_metric=primary_metric,
            timeout_minutes=timeout_minutes,
            output_database=output_database)
