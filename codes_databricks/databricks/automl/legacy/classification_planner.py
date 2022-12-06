from typing import Any, Dict

import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

from databricks.automl.legacy.const import Framework
from databricks.automl.legacy.planner import SklearnPlanner
from databricks.automl.legacy.problem_type import ProblemType
from databricks.automl.legacy.section import Section
from databricks.automl.legacy.sections.training.sklearn_classification import (
    SklearnTrainDecisionTreeClassifier, SklearnTrainLogisticRegression,
    SklearnTrainRandomForestClassifier, SklearnTrainXGBoostClassifier, SklearnTrainLGBMClassifier)


class ClassificationPlanner(SklearnPlanner):
    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.CLASSIFICATION


class SklearnLogisticRegressionTrialPlanner(ClassificationPlanner):
    _MODEL_NAME = "Logistic Regression"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainLogisticRegression

    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "sklr"
        search_space.update({
            "penalty": hp.choice(f"{prefix}_penalty", [
                {
                    "penalty": "l1",
                    "solver": "saga"
                },
                {
                    "penalty": "l2"
                },
                {
                    "penalty": "elasticnet",
                    "solver": "saga",
                    "l1_ratio": hp.loguniform(f"{prefix}_l1_ratio", np.log(1e-9), np.log(1))
                },
            ]),
            "C": hp.loguniform(f"{prefix}_C", np.log(1e-3), np.log(100)),
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False


class SklearnDecisionTreeTrialPlanner(ClassificationPlanner):
    _MODEL_NAME = "Decision Tree"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainDecisionTreeClassifier

    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "skdtc"
        search_space.update({
            "criterion": hp.choice(f"{prefix}_criterion", ["gini", "entropy"]),
            "max_depth": scope.int(hp.quniform(f"{prefix}_max_depth", 2, 12, 1)),
            "min_samples_split": hp.loguniform(f"{prefix}_min_samples_split", -3, np.log(0.3)),
            "min_samples_leaf": hp.loguniform(f"{prefix}_min_samples_leaf", -3, np.log(0.3)),
            "max_features": hp.uniform(f"{prefix}_max_features", 0.3, 0.9),
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False


class SklearnRandomForestTrialPlanner(ClassificationPlanner):
    _MODEL_NAME = "Random Forest"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainRandomForestClassifier

    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "skrfc"
        search_space.update({
            "n_estimators": scope.int(
                hp.qloguniform(f"{prefix}_n_estimators", np.log(2), np.log(2500), 1)),
            "criterion": hp.choice(f"{prefix}_criterion", ["gini", "entropy"]),
            "max_depth": scope.int(hp.quniform(f"{prefix}_max_depth", 2, 12, 1)),
            "min_samples_split": hp.uniform(f"{prefix}_min_samples_split", 0.00001, 0.5),
            "min_samples_leaf": hp.uniform(f"{prefix}_min_samples_leaf", 0.00001, 0.5),
            "bootstrap": hp.choice(f"{prefix}_bootstrap", [False, True]),
            "max_features": hp.uniform(f"{prefix}_max_features", 0.3, 0.9),
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False


class SklearnXGBoostTrialPlanner(ClassificationPlanner):
    _MODEL_NAME = "XGBoost"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainXGBoostClassifier

    @classmethod
    def framework(cls) -> Framework:
        return Framework.XGBOOST

    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "xgbc"
        search_space.update({
            "n_estimators": scope.int(
                hp.qloguniform(f"{prefix}_n_estimators", np.log(2), np.log(2500), 1)),
            "max_depth": scope.int(hp.qloguniform(f"{prefix}_max_depth", np.log(2), np.log(12), 1)),
            "learning_rate": hp.loguniform(f"{prefix}_learning_rate", np.log(1e-3), np.log(5.0)),
            "subsample": hp.loguniform(f"{prefix}_subsample", np.log(0.2), np.log(0.8)),
            "min_child_weight": scope.int(hp.quniform(f"{prefix}_min_child_weight", 1, 20, 1)),
            "colsample_bytree": hp.loguniform(f"{prefix}_colsample_bytree", np.log(0.2),
                                              np.log(0.8)),
            "n_jobs": 100,
            "verbosity": 0
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return True


class SklearnLGBMTrialPlanner(ClassificationPlanner):
    _MODEL_NAME = "LightGBM"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainLGBMClassifier

    @classmethod
    def framework(cls) -> Framework:
        return Framework.LIGHTGBM

    # Reference: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    # Base: https://github.com/microsoft/FLAML/blob/c26720c2999fa39de1f6f5581f5b52c8fb2c1f82/flaml/model.py#L188
    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "lgbmc"
        search_space.update({
            "n_estimators": scope.int(
                hp.qloguniform(f"{prefix}_n_estimators", np.log(4), np.log(2500), 1)),
            "max_depth": scope.int(hp.quniform(f"{prefix}_max_depth", 2, 12, 1)),
            "num_leaves": scope.int(
                hp.qloguniform(f"{prefix}_num_leaves", np.log(4), np.log(1024), 1)),
            "min_child_samples": scope.int(
                hp.qloguniform(f"{prefix}_min_child_samples", np.log(20), np.log(500), 1)),
            "learning_rate": hp.loguniform(f"{prefix}_learning_rate", np.log(0.01), np.log(5.0)),
            "subsample": hp.uniform(f"{prefix}_subsample", 0.5, 0.8),
            "max_bin": scope.int(hp.quniform(f"{prefix}_max_bin", 8, 500, 1)),
            "colsample_bytree": hp.uniform(f"{prefix}_colsample_bytree", 0.4, 0.8),
            "lambda_l1": hp.loguniform(f"{prefix}_lambda_l1", np.log(0.1), np.log(1024)),
            "lambda_l2": hp.loguniform(f"{prefix}_lambda_l2", np.log(0.1), np.log(1024)),
            "path_smooth": hp.uniform(f"{prefix}_path_smooth", 0, 100)
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False
