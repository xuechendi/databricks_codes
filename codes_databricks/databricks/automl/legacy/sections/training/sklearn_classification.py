from typing import List

import nbformat

from databricks.automl.legacy.problem_type import ProblemType
from databricks.automl.legacy.sections.training.sklearn import SklearnModelSection


class SklearnTrainClassifier(SklearnModelSection):
    """
    Section that uses an sklearn classifier to train a model.
    """

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.CLASSIFICATION

    def set_additional_properties(self, kwargs) -> None:
        self._multiclass = kwargs["multiclass"]

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
            multiclass=self._multiclass,
            has_datetime_columns=self._has_datetime_columns,
            parameter_dict=self._parameter_dict,
            pos_label_flag=self._pos_label_flag,
            sample_weight_col=self._sample_weight_col)


class SklearnTrainDecisionTreeClassifier(SklearnTrainClassifier):
    """
    Section that uses sklearn's DecisionTreeClassifier to train a model.
    """
    _NAME_PREFIX = "skdtc"
    _HELP_TEMPLATE = "classification/sklearn_decision_tree.help.jinja"
    _TRAINING_TEMPLATE = "classification/sklearn_decision_tree.jinja"


class SklearnTrainLogisticRegression(SklearnTrainClassifier):
    """
    Section that uses sklearn's LogisticRegression to train a model.
    """
    _NAME_PREFIX = "sklr"
    _HELP_TEMPLATE = "classification/sklearn_logistic_regression.help.jinja"
    _TRAINING_TEMPLATE = "classification/sklearn_logistic_regression.jinja"


class SklearnTrainRandomForestClassifier(SklearnTrainClassifier):
    """
    Section that uses sklearn's RandomForestClassifier to train a model.
    """
    _NAME_PREFIX = "skrf"
    _HELP_TEMPLATE = "classification/sklearn_random_forest.help.jinja"
    _TRAINING_TEMPLATE = "classification/sklearn_random_forest.jinja"


class SklearnTrainXGBoostClassifier(SklearnTrainClassifier):
    """
    Section that uses xgboost's XGBClassifier (sklearn wrapper) to train a model.
    """
    _NAME_PREFIX = "xgbc"
    _HELP_TEMPLATE = "classification/sklearn_xgboost.help.jinja"
    _TRAINING_TEMPLATE = "classification/sklearn_xgboost.jinja"


class SklearnTrainLGBMClassifier(SklearnTrainClassifier):
    """
    Section that uses lightgbm's LGBMClassifier (sklearn wrapper) to train a model.
    """
    _NAME_PREFIX = "lgbmc"
    _HELP_TEMPLATE = "classification/sklearn_lightgbm.help.jinja"
    _TRAINING_TEMPLATE = "classification/sklearn_lightgbm.jinja"
