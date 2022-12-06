from typing import List, Optional

import nbformat

from databricks.automl.internal.common.const import DatasetFormat
from databricks.automl.internal.context import DataSource
from databricks.automl.internal.sections.section import Section
from databricks.automl.shared.const import ProblemType


class LoadData(Section):
    """
    Section that loads a parquet dataset into a pandas DataFrame and split features and labels.
    """
    _LOAD_DATA_TEMPLATE = "input.load_data.jinja"

    def __init__(self,
                 var_dataframe: str,
                 data_source: DataSource,
                 load_format: DatasetFormat,
                 problem_type: ProblemType,
                 sample_fraction: Optional[float] = None,
                 unsupported_cols: Optional[List[str]] = None,
                 has_feature_store_joins: Optional[bool] = False,
                 var_feature_spec_path: Optional[str] = None,
                 name_prefix: str = "input"):
        """
        :param var_dataframe: variable name for the loaded dataframe
        :param name_prefix: name prefix for internal variables
        """
        self._var_dataframe = var_dataframe
        self._data_source = data_source
        self._load_format = load_format.value
        self._problem_type = problem_type.value
        self._sample_fraction = sample_fraction
        self._unsupported_cols = unsupported_cols or []
        self._name_prefix = name_prefix
        self._has_feature_store_joins = has_feature_store_joins
        self._var_feature_spec_path = var_feature_spec_path

        # Only pandas or pyspark.pandas is currently supported
        assert load_format in (DatasetFormat.PANDAS, DatasetFormat.PYSPARK_PANDAS)

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def input_names(self) -> List[str]:
        return []

    @property
    def output_names(self) -> List[str]:
        return [self._var_dataframe]

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        return self.template_manager.render_multicells(
            self._LOAD_DATA_TEMPLATE,
            prefix=self._name_prefix,
            var_pdf=self._var_dataframe,
            var_feature_spec_path=self._var_feature_spec_path,
            load_format=self._load_format,
            problem_type=self._problem_type,
            sample_fraction=self._sample_fraction,
            unsupported_cols=self._unsupported_cols,
            load_from_dbfs=self._data_source.is_dbfs,
            dbfs_path=self._data_source.dbfs_path,
            data_run_id=self._data_source.run_id,
            file_prefix=self._data_source.file_prefix,
            has_feature_store_joins=self._has_feature_store_joins,
        )
