from typing import List

import nbformat

from databricks.automl.legacy.section import Section


class Exit(Section):
    """
    Section that generates data that needs to be returned to the
    driver that executes this notebook
    """

    def __init__(self, var_run: str, name_prefix: str = "exp"):
        """
        :param var_run: variable name for MLflow run object
        :param name_prefix: prefix name of for this section
        """
        self._var_run = var_run
        self._name_prefix = name_prefix

    @property
    def version(self) -> str:
        return "v1"

    @property
    def return_template(self) -> str:
        return self._RETURN_TEMPLATE

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
        return_cell = self.template_manager.render_worker_only_code_cell(
            self.return_template, prefix=self._name_prefix, var_run=self._var_run)
        return [return_cell]


class NotebookExit(Exit):
    """
    Exit section for when the trials are run as notebook jobs
    """
    _RETURN_TEMPLATE = "notebook_exit.return.jinja"


class IPykernelExit(Exit):
    """
    Exit section for when the trials are run using Ipykernel
    """
    _RETURN_TEMPLATE = "ipykernel_exit.return.jinja"
