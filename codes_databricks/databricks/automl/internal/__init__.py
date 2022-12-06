import json
from typing import Any, Dict, List, Optional, Union

import mlflow
from mlflow.entities import Experiment
import pandas as pd
import pyspark.pandas as ps
import pyspark.sql

from databricks.automl.internal.common.const import ContextType
from databricks.automl.internal.classifier import Classifier
from databricks.automl.internal.confs import UnivariateForecastConf
from databricks.automl.internal.forecast import Forecast
from databricks.automl.internal.regressor import Regressor
from databricks.automl.internal.supervised_learner import SupervisedLearner
from databricks.automl.shared.const import ClassificationTargetTypes
from databricks.automl.shared.result import AutoMLSummary
from databricks.automl.shared.tags import Tag
"""
Currently, the automl/service/automl/resources/notebooks/AutoMLDriverNotebook.py directly calls
databricks.automl.classifier.Classifier(context_type=ContextType.DATABRICKS).fit(...)

We'd like to be able to modify the fit(...) internal API more easily, so in ML-25324,
we will change that notebook to call databricks.automl.internal.classify(...) instead.

This file should ONLY be called by automl/service/automl/resources/notebooks/AutoMLDriverNotebook.py

Also note that the functions in this file start with _ because otherwise, the forecast function here
is imported as `databricks.automl.internal.forecast`, which would prevent other files from importing
 `databricks.automl.internal.forecast.Forecast`
"""


def _load_experiment_tag_json(experiment: Experiment, tag: Tag) -> Optional[Any]:
    if tag in experiment.tags:
        return json.loads(experiment.tags[tag])
    return None


def _classify(table_name: str,
              *,
              target_col: str,
              primary_metric: str,
              timeout_minutes: int,
              experiment_id: str,
              data_dir: Optional[str] = None,
              exclude_cols: Optional[List[str]] = None,
              exclude_frameworks: Optional[List[str]] = None,
              experiment_dir: Optional[str] = None,
              pos_label: Optional[ClassificationTargetTypes] = None,
              time_col: Optional[str] = None) -> AutoMLSummary:
    experiment = mlflow.get_experiment(experiment_id)

    feature_store_lookups = _load_experiment_tag_json(experiment, Tag.FEATURE_STORE_LOOKUPS)
    imputers = _load_experiment_tag_json(experiment, Tag.IMPUTERS)

    return Classifier(context_type=ContextType.DATABRICKS).fit(
        dataset=table_name,
        target_col=target_col,
        data_dir=data_dir,
        exclude_cols=exclude_cols,
        exclude_frameworks=exclude_frameworks,
        home_dir=experiment_dir,
        feature_store_lookups=feature_store_lookups,
        imputers=imputers,
        metric=primary_metric,
        pos_label=pos_label,
        time_col=time_col,
        timeout_minutes=timeout_minutes,
        experiment=experiment,
    )


def _regress(table_name: str,
             *,
             target_col: str,
             primary_metric: str,
             timeout_minutes: int,
             experiment_id: str,
             data_dir: Optional[str] = None,
             exclude_cols: Optional[List[str]] = None,
             exclude_frameworks: Optional[List[str]] = None,
             experiment_dir: Optional[str] = None,
             time_col: Optional[str] = None) -> AutoMLSummary:
    experiment = mlflow.get_experiment(experiment_id)

    feature_store_lookups = _load_experiment_tag_json(experiment, Tag.FEATURE_STORE_LOOKUPS)
    imputers = _load_experiment_tag_json(experiment, Tag.IMPUTERS)

    return Regressor(context_type=ContextType.DATABRICKS).fit(
        dataset=table_name,
        target_col=target_col,
        data_dir=data_dir,
        exclude_cols=exclude_cols,
        exclude_frameworks=exclude_frameworks,
        experiment=experiment,
        home_dir=experiment_dir,
        feature_store_lookups=feature_store_lookups,
        imputers=imputers,
        metric=primary_metric,
        time_col=time_col,
        timeout_minutes=timeout_minutes)


def _forecast(table_name: str,
              *,
              target_col: str,
              primary_metric: str,
              timeout_minutes: int,
              experiment_id: str,
              time_col: str,
              frequency: str,
              horizon: int,
              data_dir: Optional[str] = None,
              exclude_frameworks: Optional[List[str]] = None,
              experiment_dir: Optional[str] = None,
              identity_col: Optional[Union[str, list]] = None,
              output_database: Optional[str] = None) -> AutoMLSummary:
    experiment = mlflow.get_experiment(experiment_id)

    return Forecast(context_type=ContextType.DATABRICKS).fit(
        dataset=table_name,
        target_col=target_col,
        time_col=time_col,
        data_dir=data_dir,
        exclude_frameworks=exclude_frameworks,
        experiment=experiment,
        home_dir=experiment_dir,
        frequency=frequency,
        horizon=horizon,
        identity_col=identity_col,
        metric=primary_metric,
        timeout_minutes=timeout_minutes,
        output_database=output_database)
