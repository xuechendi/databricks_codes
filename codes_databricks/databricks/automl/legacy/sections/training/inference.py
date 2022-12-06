from typing import List

import nbformat

from databricks.automl.legacy.section import Section


class SklearnInference(Section):
    """
    Section that shows the user how to register, load, and predict with a trained model.
    """
    _INFERENCE_TEMPLATE = "sklearn_inference.md.jinja"

    def __init__(self, var_run: str, name_prefix: str = "ski"):
        """
        :param var_run: var name of the mlflow run object
        :param name_prefix: name prefix for internal variables
        """
        self._var_run = var_run
        self._name_prefix = name_prefix

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def input_names(self) -> List[str]:
        return [self._var_run]

    @property
    def output_names(self) -> List[str]:
        return []

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        inference_cell = self.template_manager.render_multicells(
            self._INFERENCE_TEMPLATE, var_run=self._var_run)
        return inference_cell
