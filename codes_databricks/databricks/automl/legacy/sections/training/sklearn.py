from abc import abstractmethod
from typing import Any, List, Optional, Dict

import nbformat

from databricks.automl.legacy.problem_type import ClassificationTargetTypes, ProblemType
from databricks.automl.legacy.section import Section


class SklearnModelSection(Section):
    _INSTRUCTION_TEMPLATE = "sklearn_base.md.jinja"

    def __init__(self,
                 var_X_train: str,
                 var_X_val: str,
                 var_X_test: str,
                 var_y_train: str,
                 var_y_val: str,
                 var_y_test: str,
                 var_column_selector: str,
                 var_preprocessor: Optional[str],
                 var_model: str,
                 var_pipeline: str,
                 var_run: str,
                 has_datetime_columns: bool,
                 experiment_url: str,
                 experiment_id: str,
                 hyperparameters: Dict[str, Any],
                 random_state: int,
                 pos_label: Optional[ClassificationTargetTypes] = None,
                 sample_weight_col: Optional[str] = None,
                 **kwargs):
        """
        :param var_X_train: var name of training features
        :param var_X_val: var name of validation features
        :param var_X_test: var name of test features
        :param var_y_train: var name of training labels
        :param var_y_val: var name of validation labels
        :param var_y_test: var name of test labels
        :param var_column_selector: var name of column selector
        :param var_preprocessor: var name of preprocessor, or None
        :param has_datetime_columns: bool whether the dataset has datetime columns
        :param var_model: var name of model
        :param var_pipeline: var name of pipeline without the model
        :param var_run: var name of the mlflow run object
        :param experiment_url: url of the mlflow experiment used
        :param experiment_id: id of the mlflow experiment used
        :param parameter_dict: dictionary of hyperparameters to use
        :param pos_label: positive class for binary classification, or None
        :param sample_weight_col: column added by AutoML that contains sample weight
        """
        self._var_X_train = var_X_train
        self._var_X_val = var_X_val
        self._var_X_test = var_X_test
        self._var_y_train = var_y_train
        self._var_y_val = var_y_val
        self._var_y_test = var_y_test
        self._var_column_selector = var_column_selector
        self._var_preprocessor = var_preprocessor
        self._has_datetime_columns = has_datetime_columns
        self._var_model = var_model
        self._var_pipeline = var_pipeline
        self._var_run = var_run
        self._experiment_url = experiment_url
        self._experiment_id = experiment_id
        self._random_state = random_state

        self._parameter_dict = self.get_hyperparameters(hyperparameters)
        self.set_additional_properties(kwargs)

        self._pos_label_flag = ""
        if pos_label is not None:
            if isinstance(pos_label, str):
                self._pos_label_flag = f", pos_label=\"{pos_label}\""
            else:
                self._pos_label_flag = f", pos_label={pos_label}"

        self._sample_weight_col = sample_weight_col

    # Can be optionally overriden by subclasses to set additional properties based on kwargs
    def set_additional_properties(self, kwargs) -> None:
        pass

    @property
    @abstractmethod
    def problem_type(self) -> ProblemType:
        pass

    @property
    @abstractmethod
    def training_cells(self) -> List[nbformat.NotebookNode]:
        """
        Generate the code cells for training notebooks
        :return: lists of cells
        """
        pass

    def get_hyperparameters(self, hyperparameters) -> Dict:
        flat_hp = flatten(hyperparameters)

        for k, v in flat_hp.items():
            if isinstance(v, str):
                flat_hp[k] = "\"" + v + "\""

        flat_hp["random_state"] = self._random_state
        return flat_hp

    @property
    def name_prefix(self) -> str:
        return self._NAME_PREFIX

    @property
    def help_template(self) -> str:
        return self._HELP_TEMPLATE

    @property
    def training_template(self) -> str:
        return self._TRAINING_TEMPLATE

    @property
    def version(self) -> str:
        return "v1"

    @property
    def input_names(self) -> List[str]:
        inputs = [
            self._var_X_train,
            self._var_X_val,
            self._var_y_train,
            self._var_y_val,
            self._var_preprocessor,
        ]

        return list(filter(lambda x: x is not None, inputs))

    @property
    def output_names(self) -> List[str]:
        return [self._var_model, self._var_run]

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        instruction_cell = self.template_manager.render_markdown_cell(
            self._INSTRUCTION_TEMPLATE,
            problem_type=self.problem_type.value,
            experiment_url=self._experiment_url)
        help_cell = self.template_manager.render_code_cell(self.help_template)
        training_cells = self.training_cells

        return [instruction_cell, help_cell] + training_cells


def flatten(dictionary):
    flat_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            flat_dict.update(flatten(v))
        else:
            flat_dict[k] = v
    return flat_dict
