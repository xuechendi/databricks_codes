import itertools
from hashlib import blake2b
from typing import List

import nbformat

from databricks.automl.internal.sections.section import Section


class Plan:
    """
    Represents a trial plan.
    """
    _NB_METADATA_NOTEBOOK_NAME = "name"

    _NB_METADATA_DB_APPLICATION_NB_KEY = "application/vnd.databricks.v1+notebook"
    _NB_METADATA_NOTEBOOK_METADATA_KEY = "notebookMetadata"
    _NB_METADATA_EXPERIMENT_ID = "experimentId"

    def __init__(self, name: str, sections: List[Section]):
        """
        :param name: name of the trial plan
        :param sections: a list of sections to be executed
        """
        self.name = name
        self._sections = sections

    def prepend(self, section: Section):
        """
        Prepends the input section to the plan.
        """
        self._sections.insert(0, section)

    def append(self, section: Section):
        """
        Appends the input section to the plan.
        """
        self._sections.append(section)

    def __getitem__(self, index):
        return self._sections[index]

    @classmethod
    def get_nb_name(cls, notebook: nbformat.NotebookNode) -> str:
        """
        Check the notebook metadata to extract the savepath of the notebook

        :param notebook: Notebook to check
        :return: path where this notebook should be saved
        """
        return notebook.metadata.get(cls._NB_METADATA_NOTEBOOK_NAME)

    def _validate_sections(self):
        """
        Validates that all sections have required inputs provided by parent sections.
        """
        name_prefixes = set()
        names = set()
        for section in self._sections:
            assert section.name_prefix not in name_prefixes, \
                f"name_prefix {section.name_prefix} in Section {section} already exists"
            name_prefixes.add(section.name_prefix)
            for input_name in section.input_names:
                assert input_name in names, \
                    f"Section {section} is missing required input name {input_name}"
            names.update(section.output_names)

    def to_jupyter_notebook(self, experiment_id: str) -> nbformat.NotebookNode:
        """
        Converts to plan to a Jupyter notebook in nbformat.
        :param name: unique name of the notebook
        :param experiment_id: The MLflow experiment id of the associated experiment

        :returns notebook for the plan
        """
        self._validate_sections()
        cells = list(itertools.chain.from_iterable([s.cells for s in self._sections]))
        nb = nbformat.v4.new_notebook()
        nb["cells"] = cells

        # generate a hash of the notebook and store it as metadata
        param_hash = blake2b(digest_size=16)
        param_hash.update(str(nb).encode("utf-8"))

        nb.metadata[self._NB_METADATA_NOTEBOOK_NAME] = f"{self.name}-{param_hash.hexdigest()}"

        # setting the experiment id here will link it to the notebook sidebar
        nb.metadata[self._NB_METADATA_DB_APPLICATION_NB_KEY] = {
            self._NB_METADATA_NOTEBOOK_METADATA_KEY: {
                self._NB_METADATA_EXPERIMENT_ID: experiment_id
            }
        }

        return nb
