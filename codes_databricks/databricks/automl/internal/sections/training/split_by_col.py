from typing import List, Optional

import nbformat

from databricks.automl.internal.sections.section import Section


class SplitDataByCol(Section):
    _CODE_TEMPLATE = "split_by_col.jinja"

    @property
    def version(self) -> str:
        return "v1"

    def __init__(self,
                 var_input_df: str,
                 split_col: str,
                 var_target_col: str,
                 var_X_train: str,
                 var_X_val: str,
                 var_X_test: str,
                 var_y_train: str,
                 var_y_val: str,
                 var_y_test: str,
                 var_sample_weight: str,
                 sample_weight_col: str,
                 time_col: Optional[str] = None,
                 name_prefix: str = "split"):
        """
        :param var_input_df: variable name of input dataframe
        :param split_col: column added by AutoML that specifies the split ("train" / "val" / "test")
        :param var_target_col: var name of target column
        :param var_X_train: var name of training features
        :param var_X_val: var name of validation features
        :param var_y_train: var name of training labels
        :param var_y_val: var name of validation labels
        :param time_col: optional time column for splitting by time
        """
        self._var_input_df = var_input_df
        self._split_col = split_col
        self._var_target_col = var_target_col
        self._var_X_train = var_X_train
        self._var_X_val = var_X_val
        self._var_X_test = var_X_test
        self._var_y_train = var_y_train
        self._var_y_val = var_y_val
        self._var_y_test = var_y_test
        self._var_sample_weight = var_sample_weight
        self._sample_weight_col = sample_weight_col
        self._time_col = time_col
        self._name_prefix = name_prefix

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def input_names(self) -> List[str]:
        return [self._var_input_df, self._var_target_col]

    @property
    def output_names(self) -> List[str]:
        return [
            self._var_X_train,
            self._var_X_val,
            self._var_y_train,
            self._var_y_val,
        ]

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        cell = self.template_manager.render_multicells(
            self._CODE_TEMPLATE,
            var_input_df=self._var_input_df,
            split_col=self._split_col,
            var_target_col=self._var_target_col,
            var_X_train=self._var_X_train,
            var_X_val=self._var_X_val,
            var_X_test=self._var_X_test,
            var_y_train=self._var_y_train,
            var_y_val=self._var_y_val,
            var_y_test=self._var_y_test,
            var_sample_weight=self._var_sample_weight,
            sample_weight_col=self._sample_weight_col,
            time_col=self._time_col,
            prefix=self._name_prefix,
        )
        return cell
