from typing import List

import nbformat

from databricks.automl.legacy.problem_type import ProblemType
from databricks.automl.legacy.section import Section


class ShapFeatureImportancePlot(Section):
    """
    Section that uses shap's summary_plot to show feature importance of a trained model.
    """

    _MAXIMUM_BACKGROUND_DATA_SIZE = 100  # matches https://github.com/mlflow/mlflow/blob/master/mlflow/shap.py#L12

    _SHAP_PLOT_TEMPLATE = "shap_plot.jinja"

    def __init__(self,
                 var_X_train: str,
                 var_X_val: str,
                 var_model: str,
                 use_predict_proba: bool,
                 has_nulls: bool,
                 problem_type: ProblemType,
                 random_state: int,
                 name_prefix: str = "sfi"):
        """
        :param var_X_train: var name of training features
        :param var_X_val: var name of validation features
        :param var_model: var name of model
        :param use_predict_proba: whether to use predict or predict_proba on model
        :param has_nulls: whether the dataset has any nulls
        :param problem_type: problem type
        """
        self._var_X_train = var_X_train
        self._var_X_val = var_X_val
        self._var_model = var_model
        self._use_predict_proba = use_predict_proba
        self._has_nulls = has_nulls
        self._problem_type = problem_type
        self._random_state = random_state
        self._name_prefix = name_prefix

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def input_names(self) -> List[str]:
        return [self._var_X_train, self._var_X_val, self._var_model]

    @property
    def output_names(self) -> List[str]:
        return []

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        shap_plot_cells = self.template_manager.render_multicells(
            self._SHAP_PLOT_TEMPLATE,
            maximum_background_data_size=self._MAXIMUM_BACKGROUND_DATA_SIZE,
            use_predict_proba=self._use_predict_proba,
            has_nulls=self._has_nulls,
            problem_type=self._problem_type.value,
            var_X_train=self._var_X_train,
            var_X_val=self._var_X_val,
            var_model=self._var_model,
            random_state=self._random_state)

        return shap_plot_cells
