from typing import List

import nbformat

from databricks.automl.internal.sections.section import Section


class DatabricksPredefined(Section):
    """
    Dummy section that does not generate any cells but lists the output
    variables that can be assumed to be predefined in the given notebook
    """

    def __init__(self, api_url, token):
        self._api_url = api_url
        self._token = token
        pass

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return ""

    @property
    def input_names(self) -> List[str]:
        return []

    @property
    def output_names(self) -> List[str]:
        return ["spark"]

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        conf_cell = self.template_manager.render_worker_only_code_cell(
            "prepend.databricks.jinja", host=self._api_url, token=self._token)
        return [conf_cell]
