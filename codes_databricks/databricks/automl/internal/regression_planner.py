from typing import Any, Dict

import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

from databricks.automl.internal.common.const import Framework
from databricks.automl.internal.planner import SklearnPlanner
from databricks.automl.internal.sections.section import Section
from databricks.automl.internal.sections.training.sklearn_regression import SklearnTrainSGDRegressor, \
    SklearnTrainDecisionTreeRegressor, SklearnTrainRandomForestRegressor, SklearnTrainXGBoostRegressor, \
    SklearnTrainLGBMRegressor
from databricks.automl.shared.const import ProblemType


class RegressionPlanner(SklearnPlanner):
    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.REGRESSION


class SGDRegressionPlanner(RegressionPlanner):
    _MODEL_NAME = "SGD Regression"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainSGDRegressor

    # https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/regression/sgd.py
    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "sksgd"
        epsilon_space = hp.loguniform(f"{prefix}_epsilon", np.log(1e-5), np.log(1e-1))
        eta0_space = hp.loguniform(f"{prefix}_eta0", np.log(1e-5), np.log(1e-1))
        search_space.update({
            "fit_intercept": True,
            "alpha": hp.loguniform(f"{prefix}_alpha", np.log(1e-7), np.log(1e-1)),
            "tol": hp.loguniform(f"{prefix}_tol", np.log(1e-5), np.log(1e-1)),
            "average": hp.choice(f"{prefix}_average", [False, True]),
            "loss": hp.choice(f"{prefix}_loss", [
                {
                    "loss": "squared_loss"
                },
                {
                    "loss": "huber",
                    "epsilon": epsilon_space
                },
                {
                    "loss": "squared_epsilon_insensitive",
                    "epsilon": epsilon_space
                },
            ]),
            "penalty": hp.choice(f"{prefix}_penalty", [
                {
                    "penalty": "l1"
                },
                {
                    "penalty": "l2"
                },
                {
                    "penalty": "elasticnet",
                    "l1_ratio": hp.loguniform(f"{prefix}_l1_ratio", np.log(1e-9), np.log(1))
                },
            ]),
            "learning_rate": hp.choice(f"{prefix}_learning_rate", [
                {
                    "learning_rate": "invscaling",
                    "eta0": eta0_space,
                    "power_t": hp.loguniform(f"{prefix}_power_t", np.log(1e-5), np.log(1))
                },
                {
                    "learning_rate": "adaptive",
                    "eta0": eta0_space
                },
            ]),
            # defaults from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
            "early_stopping": False,
            "n_iter_no_change": 5,
            "validation_fraction": 0.1,
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False


class DecisionTreeRegressorPlanner(RegressionPlanner):
    _MODEL_NAME = "Decision Tree Regressor"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainDecisionTreeRegressor

    # https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/regression/decision_tree.py
    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "skdt"
        search_space.update({
            "criterion": hp.choice(f"{prefix}_criterion", ["mse", "friedman_mse", "mae"]),
            "max_depth": scope.int(hp.quniform(f"{prefix}_max_depth", 2, 12, 1)),
            "max_features": hp.uniform(f"{prefix}_max_features", 0.3, 0.9),
            "min_samples_split": hp.loguniform(f"{prefix}_min_samples_split", -3, np.log(0.3)),
            "min_samples_leaf": hp.loguniform(f"{prefix}_min_samples_leaf", -3, np.log(0.3)),
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False


class RandomForestRegressorPlanner(RegressionPlanner):
    _MODEL_NAME = "Random Forest Regressor"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainRandomForestRegressor

    # https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/regression/random_forest.py
    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "skrf"
        search_space.update({
            "n_estimators": scope.int(
                hp.qloguniform(f"{prefix}_n_estimators", np.log(2), np.log(2500), 1)),
            "criterion": hp.choice(f"{prefix}_criterion", ["mse", "friedman_mse", "mae"]),
            "max_depth": scope.int(hp.quniform(f"{prefix}_max_depth", 2, 12, 1)),
            "max_features": hp.uniform(f"{prefix}_max_features", 0.3, 0.9),
            "min_samples_split": hp.uniform(f"{prefix}_min_samples_split", 0.00001, 0.5),
            "min_samples_leaf": hp.uniform(f"{prefix}_min_samples_leaf", 0.00001, 0.5),
            "bootstrap": hp.choice(f"{prefix}_bootstrap", [True, False])
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False


class XGBoostRegressorPlanner(RegressionPlanner):
    _MODEL_NAME = "XGBoost Regressor"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainXGBoostRegressor

    @classmethod
    def framework(cls) -> Framework:
        return Framework.XGBOOST

    # https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor.py#L98
    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "xgbr"
        search_space.update({
            "n_estimators": scope.int(
                hp.qloguniform(f"{prefix}_n_estimators", np.log(2), np.log(2500), 1)),
            "max_depth": scope.int(hp.qloguniform(f"{prefix}_max_depth", np.log(2), np.log(12), 1)),
            "learning_rate": hp.loguniform(f"{prefix}_learning_rate", np.log(1e-3), np.log(5.0)),
            "subsample": hp.uniform(f"{prefix}_subsample", 0.2, 0.8),
            "min_child_weight": scope.int(hp.quniform(f"{prefix}_min_child_weight", 1, 20, 1)),
            "colsample_bytree": hp.uniform(f"{prefix}_colsample_bytree", 0.2, 0.8),
            "n_jobs": 100,
            "verbosity": 0
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False


class LGBMRegressorPlanner(RegressionPlanner):
    _MODEL_NAME = "LightGBM"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def model_class(self) -> Section:
        return SklearnTrainLGBMRegressor

    @classmethod
    def framework(cls) -> Framework:
        return Framework.LIGHTGBM

    # https://github.com/microsoft/FLAML/blob/main/flaml/model.py#L188
    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        search_space = super().get_hyperparameter_search_space()
        prefix = "lgbmr"
        search_space.update({
            "n_estimators": scope.int(
                hp.qloguniform(f"{prefix}_n_estimators", np.log(4), np.log(2500), 1)),
            "max_depth": scope.int(hp.qloguniform(f"{prefix}_max_depth", np.log(2), np.log(12), 1)),
            "num_leaves": scope.int(
                hp.qloguniform(f"{prefix}_num_leaves", np.log(4), np.log(1024), 1)),
            "min_child_samples": scope.int(
                hp.qloguniform(f"{prefix}_min_child_samples", np.log(20), np.log(500), 1)),
            "learning_rate": hp.loguniform(f"{prefix}_learning_rate", np.log(0.01), np.log(5.0)),
            "subsample": hp.uniform(f"{prefix}_subsample", 0.5, 0.8),
            "max_bin": scope.int(hp.quniform(f"{prefix}_max_bin", 8, 500, 1)),
            "colsample_bytree": hp.uniform(f"{prefix}_colsample_bytree", 0.4, 0.8),
            "lambda_l1": hp.loguniform(f"{prefix}_lambda_l1", np.log(0.1), np.log(1024)),
            "lambda_l2": hp.loguniform(f"{prefix}_lambda_l2", np.log(0.1), np.log(1024))
        })
        return search_space

    @staticmethod
    def requires_data_imputation() -> bool:
        return False
