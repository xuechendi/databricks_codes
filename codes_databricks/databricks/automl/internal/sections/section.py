from abc import ABC, abstractmethod
from typing import List

import nbformat

from databricks.automl.internal.sections.template import SectionTemplateManager


class Section(ABC):
    """
    Represents a functional section in a notebook that contains code and markdown cells.
    """
    template_manager = SectionTemplateManager()

    @property
    @abstractmethod
    def version(self) -> str:
        """
        Version of the code section.
        """
        pass

    @property
    @abstractmethod
    def name_prefix(self) -> str:
        """
        Prefix for all internal variable names to avoid naming conflicts.
        """
        pass

    @property
    @abstractmethod
    def input_names(self) -> List[str]:
        """
        Expected input variable and method names.
        """
        pass

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        """
        Output variable and method names to be used by sections.
        """
        pass

    @property
    @abstractmethod
    def cells(self) -> List[nbformat.NotebookNode]:
        """
        A list of generated cells in nbformat NotebookNode.
        """
        pass

    def __str__(self):
        return self.__class__.__name__
