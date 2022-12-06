from typing import List

import nbformat

from databricks.automl.internal.sections.section import Section


class JupyterDefinitions(Section):
    """
    Section that contains common definitions needed by Jupyter notebook execution (test only).
    """

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return "jupyter"

    @property
    def input_names(self) -> List[str]:
        return []

    @property
    def output_names(self) -> List[str]:
        return ["spark"]

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        spark_dbutils_cell = self.template_manager.render_code_cell("jupyter.jinja")

        return [spark_dbutils_cell]
