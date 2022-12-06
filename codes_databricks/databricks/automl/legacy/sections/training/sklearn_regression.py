from typing import List

import nbformat

from databricks.automl.legacy.problem_type import ProblemType
from databricks.automl.legacy.sections.training.sklearn import SklearnModelSection


class SklearnTrainRegressor(SklearnModelSection):
    """
    Section that uses an sklearn regressor to train a model.
    """

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.REGRESSION

    @property
    def training_cells(self) -> List[nbformat.NotebookNode]:
        return self.template_manager.render_multicells(
            self.training_template,
            prefix=self.name_prefix,
            var_column_selector=self._var_column_selector,
            var_preprocessor=self._var_preprocessor,
            var_model=self._var_model,
            var_pipeline=self._var_pipeline,
            var_run=self._var_run,
            var_X_train=self._var_X_train,
            var_y_train=self._var_y_train,
            var_X_val=self._var_X_val,
            var_y_val=self._var_y_val,
            var_X_test=self._var_X_test,
            var_y_test=self._var_y_test,
            experiment_id=self._experiment_id,
            has_datetime_columns=self._has_datetime_columns,
            parameter_dict=self._parameter_dict)


class SklearnTrainSGDRegressor(SklearnTrainRegressor):
    """
    Section that uses sklearn's SGDRegressor to train a model.
    """
    _NAME_PREFIX = "sgdr"
    _HELP_TEMPLATE = "regression/sklearn_sgd_regression.help.jinja"
    _TRAINING_TEMPLATE = "regression/sklearn_sgd_regression.jinja"


class SklearnTrainDecisionTreeRegressor(SklearnTrainRegressor):
    """
    Section that uses sklearn's DecisionTreeRegressor to train a model.
    """
    _NAME_PREFIX = "skdt"
    _HELP_TEMPLATE = "regression/sklearn_decision_tree.help.jinja"
    _TRAINING_TEMPLATE = "regression/sklearn_decision_tree.jinja"


class SklearnTrainRandomForestRegressor(SklearnTrainRegressor):
    """
    Section that uses sklearn's RandomForestRegressor to train a model.
    """
    _NAME_PREFIX = "skrf"
    _HELP_TEMPLATE = "regression/sklearn_random_forest.help.jinja"
    _TRAINING_TEMPLATE = "regression/sklearn_random_forest.jinja"


class SklearnTrainXGBoostRegressor(SklearnTrainRegressor):
    """
    Section that uses XGBoost to train a model.
    """
    _NAME_PREFIX = "xgb"
    _HELP_TEMPLATE = "regression/sklearn_xgboost.help.jinja"
    _TRAINING_TEMPLATE = "regression/sklearn_xgboost.jinja"


class SklearnTrainLGBMRegressor(SklearnTrainRegressor):
    """
    Section that uses lightgbm's LGBMRegressor to train a model.
    """
    _NAME_PREFIX = "lgbmr"
    _HELP_TEMPLATE = "regression/sklearn_lightgbm.help.jinja"
    _TRAINING_TEMPLATE = "regression/sklearn_lightgbm.jinja"
