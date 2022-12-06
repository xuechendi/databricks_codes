from typing import List

import nbformat

from databricks.automl.internal.sections.section import Section


class ClassificationEvaluationPlots(Section):
    """
    Section that shows MLflow run artifact plots for classification, including confusion matrix, ROC and PR curves.
    """
    _EVAL_PLOTS_TEMPLATE = "classification/eval_plots.jinja"

    def __init__(self, experiment_id: str, var_run: str, multiclass: bool):
        """
        :param experiment_id: id of mlflow experiment
        :param var_run: var name of the mlflow run object
        :param multiclass: whether the classification problem is multiclass
        """
        self._experiment_id = experiment_id
        self._var_run = var_run
        self._multiclass = multiclass

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return "eval"

    @property
    def input_names(self) -> List[str]:
        return [self._var_run]

    @property
    def output_names(self) -> List[str]:
        return []

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        eval_plot_cells = self.template_manager.render_multicells(
            self._EVAL_PLOTS_TEMPLATE,
            prefix=self.name_prefix,
            experiment_id=self._experiment_id,
            var_run=self._var_run,
            multiclass=self._multiclass)

        return eval_plot_cells
