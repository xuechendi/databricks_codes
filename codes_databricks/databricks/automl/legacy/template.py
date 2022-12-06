from typing import Optional, List

import nbformat
from jinja2 import Environment, PackageLoader, StrictUndefined, FileSystemLoader, BaseLoader


class TemplateManager:
    # To split template with multiple cells
    _MULTICELL_DELIMITER = "# COMMAND ----------"
    _MARKDOWN_TAG = "%md"
    _LARGE_DISPLAY_OUTPUT_TAG = "%large-display-output"

    def __init__(self, loader: BaseLoader):
        """
        Class that manages jinja templates and their adds helper functions
        for rendering them
        :param loader: The filesystem or package loader to be used to load the templates
        """
        self._env = Environment(loader=loader, trim_blocks=True, undefined=StrictUndefined)
        self._env = self._initialize_filters(self._env)

    @staticmethod
    def get_package_loader(package: str):
        return PackageLoader(package, "templates")

    @staticmethod
    def get_filesystem_loader(location: str):
        return FileSystemLoader(location)

    @staticmethod
    def _initialize_filters(env):
        def camelcase(name: str) -> str:
            tokens = name.split(" ")
            tokens = [t.lower() for t in tokens if t != ""]
            return "_".join(tokens)

        def titlecase(name: str) -> str:
            tokens = name.split(" ")
            tokens = [t.capitalize() for t in tokens if t != ""]
            return " ".join(tokens)

        env.filters["camelcase"] = camelcase
        env.filters["titlecase"] = titlecase
        return env

    def render(self, template_name: str, **kwargs) -> str:
        """
        Renders the template using the arguments provided

        :param template_name: Name of the template to be rendered
        :param kwargs:        Additional key value pairs to pass to the template when rendered
        :return:              Rendered template string
        """
        template = self._env.get_template(template_name)
        return template.render(**kwargs).strip()

    def render_code_cell(self, template_name: str, metadata: Optional[dict] = None,
                         **kwargs) -> nbformat.NotebookNode:
        """
        Renders a code cell using the given template and the arguments provided

        :param template_name: Name of the template to be rendered
        :param metadata: metadata for the code cell
        :param kwargs:        Additional key value pairs to pass to the template when rendered
        :return:              notebook code cell
        """
        return nbformat.v4.new_code_cell(
            source=self.render(template_name, **kwargs), metadata=metadata or {})

    def render_worker_only_code_cell(self, template_name: str, **kwargs) -> nbformat.NotebookNode:
        return self.render_code_cell(template_name, metadata={"worker_config_only": True}, **kwargs)

    def render_markdown_cell(self, template_name: str, metadata: Optional[dict] = None,
                             **kwargs) -> nbformat.NotebookNode:
        """
        Renders a markdown cell using the given template and the arguments provided

        :param template_name: Name of the template to be rendered
        :param metadata: metadata for the code cell
        :param kwargs:        Additional key value pairs to pass to the template when rendered
        :return:              notebook markdown cell
        """
        return nbformat.v4.new_markdown_cell(
            source=self.render(template_name, **kwargs), metadata=metadata or {})

    def render_multicells(self, template_name: str, metadata: Optional[dict] = None,
                          **kwargs) -> List[nbformat.NotebookNode]:
        """
        Renders the template with multiple cells and split them into code / markdown cells

        :param template_name: Name of the template to be rendered
        :param metadata: metadata for all the cells rendered
        :param kwargs: Additional key value pairs to pass to the template when rendering the template
        :return: List of cells rendered
        """
        raw_notebook = self.render(template_name, **kwargs)
        raw_cells = raw_notebook.split(self._MULTICELL_DELIMITER)
        cells = []
        for content in raw_cells:
            content = content.strip()
            if content.startswith(self._MARKDOWN_TAG):
                content = content.replace(self._MARKDOWN_TAG, "")
                cells.append(
                    nbformat.v4.new_markdown_cell(source=content.strip(), metadata=metadata or {}))
            else:
                cell_metadata = metadata.copy() if metadata else {}
                if content.startswith(self._LARGE_DISPLAY_OUTPUT_TAG):
                    content = content.replace(self._LARGE_DISPLAY_OUTPUT_TAG, "")
                    cell_metadata["large_display_output"] = True

                cells.append(
                    nbformat.v4.new_code_cell(source=content.strip(), metadata=cell_metadata or {}))
        return cells


class SectionTemplateManager(TemplateManager):
    """
    Template manager for automl sections
    """
    _SECTIONS_PACKAGE = "databricks.automl.legacy.sections"

    def __init__(self):
        package_loader = TemplateManager.get_package_loader(self._SECTIONS_PACKAGE)
        super().__init__(loader=package_loader)
