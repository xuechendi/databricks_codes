from typing import Dict, List, Optional, Set

import nbformat
from pyspark.sql.types import StringType

from databricks.automl.legacy.const import AutoMLDataType, SparseOrDense
from databricks.automl.legacy.errors import InvalidSectionInputError
from databricks.automl.legacy.imputers import ImputeConstant, Imputer, ImputeMean
from databricks.automl.legacy.section import Section


class PreprocessSetup(Section):
    _CODE_TEMPLATE = "preprocess/setup.jinja"

    def __init__(self, name_prefix: str = "ps"):
        self._name_prefix = name_prefix

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
        return []

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        setup_cells = self.template_manager.render_multicells(self._CODE_TEMPLATE)
        return setup_cells


class PreprocessFinish(Section):
    _CODE_TEMPLATE = "preprocess/finish.jinja"

    def __init__(self,
                 transformer_lists_used: List[str],
                 var_preprocessor: str,
                 sparse_or_dense: SparseOrDense,
                 name_prefix: str = "pf"):
        self._transformer_lists_used = transformer_lists_used
        self._var_preprocessor = var_preprocessor
        self._name_prefix = name_prefix
        if sparse_or_dense == SparseOrDense.SPARSE:
            self._sparse_threshold = 1
        else:
            self._sparse_threshold = 0

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def input_names(self) -> List[str]:
        return self._transformer_lists_used

    @property
    def output_names(self) -> List[str]:
        return [self._var_preprocessor]

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        code_cell = self.template_manager.render_code_cell(
            self._CODE_TEMPLATE,
            transformer_lists_used=self._transformer_lists_used,
            var_preprocessor=self._var_preprocessor,
            sparse_threshold=self._sparse_threshold)
        return [code_cell]


class Preprocessor(Section):
    """
    Base class for all preprocessors.
    """

    def __init__(self, name_prefix: str, transformer_output_names: List[str] = []):
        self._name_prefix = name_prefix
        self._transformer_output_names = transformer_output_names

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
        return self._transformer_output_names

    @property
    def transformer_output_names(self) -> List[str]:
        """
        The variable names of lists of transformers defined by this preprocessor.
        """
        return self._transformer_output_names


class DatetimePreprocessor(Preprocessor):
    """
    Creates preprocessors for datetime columns.
    """
    _MARKDOWN_TEMPLATE = "preprocess/datetime.md.jinja"
    _DATE_TEMPLATE = "preprocess/datetime_date.jinja"
    _TIMESTAMP_TEMPLATE = "preprocess/datetime_timestamp.jinja"

    def __init__(self,
                 date_columns: Set[str],
                 timestamp_columns: Set[str],
                 imputers: Dict[str, Imputer],
                 var_date_transformers: str,
                 var_datetime_transformers: str,
                 name_prefix: str = "dt"):
        transformer_output_names = []
        if date_columns:
            transformer_output_names.append(var_date_transformers)
        if timestamp_columns:
            transformer_output_names.append(var_datetime_transformers)

        super(DatetimePreprocessor, self).__init__(
            name_prefix, transformer_output_names=transformer_output_names)

        if not date_columns and not timestamp_columns:
            raise InvalidSectionInputError(
                "Both date_columns and timestamp_columns are empty or None")
        self._date_columns = date_columns
        self._timestamp_columns = timestamp_columns
        self._var_date_transformers = var_date_transformers
        self._var_datetime_transformers = var_datetime_transformers

        self._date_columns = sorted(date_columns)
        self._timestamp_columns = sorted(timestamp_columns)

        def get_imputer(col):
            if col in imputers:
                return imputers[col]
            else:
                return ImputeMean(AutoMLDataType.DATETIME)

        self._date_columns_and_imputers = {col: get_imputer(col) for col in self._date_columns}
        self._timestamp_columns_and_imputers = {
            col: get_imputer(col)
            for col in self._timestamp_columns
        }

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        cells = [
            self.template_manager.render_markdown_cell(
                self._MARKDOWN_TEMPLATE, prefix=self.name_prefix)
        ]

        if self._date_columns:
            date_cell = self.template_manager.render_code_cell(
                self._DATE_TEMPLATE,
                date_columns_and_imputers=self._date_columns_and_imputers,
                var_date_transformers=self._var_date_transformers,
                prefix=self.name_prefix)
            cells.append(date_cell)

        if self._timestamp_columns:
            timestamp_cell = self.template_manager.render_code_cell(
                self._TIMESTAMP_TEMPLATE,
                timestamp_columns_and_imputers=self._timestamp_columns_and_imputers,
                var_datetime_transformers=self._var_datetime_transformers,
                prefix=self.name_prefix)
            cells.append(timestamp_cell)

        return cells


class CategoricalPreprocessor(Preprocessor):
    """
    Create preprocessors for string columns, using one-hot encoding or feature hashing.
    """

    # Default number of outputs columns generated per input column by feature hashing
    DEFAULT_HASH_OUTPUT_COLS = 1024

    _CATEGORICAL_MARKDOWN_TEMPLATE = "preprocess/categorical.md.jinja"
    _ONE_HOT_TEMPLATE = "preprocess/categorical_one_hot.jinja"
    _HASH_TEMPLATE = "preprocess/categorical_hash.jinja"

    def __init__(self,
                 var_categorical_hash_transformers: str,
                 var_categorical_one_hot_transformers: str,
                 imputers: Dict[str, Imputer],
                 one_hot_cols: Optional[Set[str]] = None,
                 hash_cols: Optional[Set[str]] = None,
                 num_hash_output_cols: int = DEFAULT_HASH_OUTPUT_COLS,
                 name_prefix: str = "cp"):
        """
        :param var_categorical_hash_transformers: var name of categorical_hash_transformers
        :param var_categorical_one_hot_transformers: var name of categorical_one_hot_transformers
        :param imputers: dictionary where keys are column names and values are imputation strategies
        :param one_hot_cols: columns to apply one-hot encoding
        :param hash_cols: string columns to apply feature hashing
        :param num_hash_output_cols: number of output columns for each feature-hashed input column
        """
        transformer_output_names = []
        if one_hot_cols:
            transformer_output_names.append(var_categorical_one_hot_transformers)
        if hash_cols:
            transformer_output_names.append(var_categorical_hash_transformers)

        super(CategoricalPreprocessor, self).__init__(
            name_prefix, transformer_output_names=transformer_output_names)

        if not one_hot_cols and not hash_cols:
            raise InvalidSectionInputError("one_hot_cols and hash_cols are empty or None")
        self._one_hot_cols = sorted(one_hot_cols or [])
        self._hash_cols = sorted(hash_cols or [])
        self._num_hash_output_cols = num_hash_output_cols
        self._var_categorical_hash_transformers = var_categorical_hash_transformers
        self._var_categorical_one_hot_transformers = var_categorical_one_hot_transformers

        one_hot_imputers = {}
        for col, imputer in imputers.items():
            if col in self._one_hot_cols:
                one_hot_imputers.setdefault(imputer, []).append(col)

        self._one_hot_imputers = []
        for imputer, cols in one_hot_imputers.items():
            self._one_hot_imputers.append(
                [cols, imputer.get_name(),
                 repr(imputer.get_sklearn_imputer())])

        self._hash_imputers = []
        for col, imputer in imputers.items():
            if col in self._hash_cols:
                self._hash_imputers.append([
                    col,
                    imputer.get_name(),
                    repr(imputer.get_sklearn_imputer()),
                ])
        self._default_hash_imputer = ImputeConstant(spark_type=StringType(), fill_value="")

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        cells = [self.template_manager.render_markdown_cell(self._CATEGORICAL_MARKDOWN_TEMPLATE)]
        if self._one_hot_cols:
            one_hot_cells = self.template_manager.render_multicells(
                self._ONE_HOT_TEMPLATE,
                one_hot_cols=self._one_hot_cols,
                var_categorical_one_hot_transformers=self._var_categorical_one_hot_transformers,
                imputers=self._one_hot_imputers,
            )
            cells += one_hot_cells
        if self._hash_cols:
            hash_cells = self.template_manager.render_multicells(
                self._HASH_TEMPLATE,
                hash_cols=self._hash_cols,
                num_hash_output_cols=self._num_hash_output_cols,
                var_categorical_hash_transformers=self._var_categorical_hash_transformers,
                imputers=self._hash_imputers,
                default_imputer_name=self._default_hash_imputer.get_name(),
                default_imputer=self._default_hash_imputer.get_sklearn_imputer(),
            )
            cells += hash_cells

        return cells


class ColumnSelectorPreprocessor(Preprocessor):
    """
    Create preprocessors to select the supported columns
    """

    _CODE_TEMPLATE = "preprocess/select_columns.jinja"

    def __init__(self,
                 var_column_selector: str,
                 var_supported_cols: str,
                 unsupported_cols: Set[str],
                 supported_cols: Set[str],
                 name_prefix: str = "sel"):
        """
        :param var_column_selector: var name of the column selector transformer
        :param var_supported_cols: var name of the list of the supported columns
        :param supported_cols: supported columns
        """
        super(ColumnSelectorPreprocessor, self).__init__(name_prefix)

        if not supported_cols:
            raise InvalidSectionInputError("supported_cols is empty or None")
        self._var_column_selector = var_column_selector
        self._var_supported_cols = var_supported_cols
        self._unsupported_cols = unsupported_cols
        self._supported_cols = supported_cols

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        preprocess_cells = self.template_manager.render_multicells(
            self._CODE_TEMPLATE,
            var_column_selector=self._var_column_selector,
            var_supported_cols=self._var_supported_cols,
            unsupported_cols=self._unsupported_cols,
            supported_cols=self._supported_cols)
        return preprocess_cells


class BooleanPreprocessor(Preprocessor):
    """
    Create preprocessors to boolean columns
    """

    _CODE_TEMPLATE = "preprocess/boolean.jinja"

    def __init__(self,
                 var_bool_transformers: str,
                 imputers: Dict[str, Imputer],
                 boolean_cols: Set[str],
                 name_prefix: str = "bp"):
        """
        :param var_bool_transformers: var name of bool_transformers
        :param imputers: dictionary where keys are column names and values are imputation strategies
        :param boolean_cols: boolean columns
        """
        super(BooleanPreprocessor, self).__init__(
            name_prefix, transformer_output_names=[var_bool_transformers])

        if not boolean_cols:
            raise InvalidSectionInputError("boolean_cols is empty or None")
        self._boolean_cols = boolean_cols
        self._var_bool_transformers = var_bool_transformers

        # Create a map from imputer to columns, where the same kind of imputers will be grouped together.
        imputer_to_cols = {}
        cols_with_custom_imputer = set()
        for col, imputer in imputers.items():
            if col in boolean_cols:
                imputer_to_cols.setdefault(imputer, []).append(col)
                cols_with_custom_imputer.add(col)

        self._imputers = []
        for imputer, cols in imputer_to_cols.items():
            self._imputers.append(
                [sorted(cols),
                 imputer.get_name(),
                 repr(imputer.get_sklearn_imputer())])

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        preprocess_cells = self.template_manager.render_multicells(
            self._CODE_TEMPLATE,
            boolean_cols=self._boolean_cols,
            imputers=self._imputers,
            var_bool_transformers=self._var_bool_transformers)
        return preprocess_cells


class NumericalPreprocessor(Preprocessor):
    """
    Create preprocessors to numerical columns
    """

    _CODE_TEMPLATE = "preprocess/numerical.jinja"

    def __init__(self,
                 var_numerical_transformers: str,
                 numerical_cols: Set[str],
                 imputers: Dict[str, Imputer],
                 name_prefix: str = "np"):
        """
        :param var_numerical_transformers: var name of numerical_transformers
        :param numerical_cols: numerical columns
        """
        super(NumericalPreprocessor, self).__init__(
            name_prefix, transformer_output_names=[var_numerical_transformers])

        if not numerical_cols:
            raise InvalidSectionInputError("numerical_cols is empty or None")
        self._numerical_cols = numerical_cols
        self._var_numerical_transformers = var_numerical_transformers

        imputers_to_cols = {}
        for col in numerical_cols:
            # Default imputation is mean
            imputer = imputers.get(col, ImputeMean(AutoMLDataType.NUMERIC))
            imputers_to_cols.setdefault(imputer, []).append(col)
        for imputer, cols in imputers_to_cols.items():
            imputers_to_cols[imputer] = sorted(cols)
        self._imputers = imputers_to_cols

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        preprocess_cells = self.template_manager.render_multicells(
            self._CODE_TEMPLATE,
            numerical_cols=self._numerical_cols,
            imputers=self._imputers,
            var_numerical_transformers=self._var_numerical_transformers)
        return preprocess_cells


class ArrayPreprocessor(Preprocessor):
    """
    Create preprocessors to array columns
    """

    _CODE_TEMPLATE = "preprocess/array.jinja"

    def __init__(self,
                 var_array_transformers: str,
                 array_cols: Set[str] = None,
                 name_prefix: str = "array"):
        """
        :param var_array_transformers: var name of array_transformers
        :param array_cols: array columns
        """
        super(ArrayPreprocessor, self).__init__(
            name_prefix, transformer_output_names=[var_array_transformers])

        if not array_cols:
            raise InvalidSectionInputError("array_cols is empty or None")
        self._array_cols = array_cols
        self._var_array_transformers = var_array_transformers

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        preprocess_cells = self.template_manager.render_multicells(
            self._CODE_TEMPLATE,
            prefix=self.name_prefix,
            array_cols=self._array_cols,
            var_array_transformers=self._var_array_transformers)
        return preprocess_cells


class TextPreprocessor(Preprocessor):
    """
    Create preprocessors for text columns.
    """

    # Default number of outputs columns generated per input column by tf-idf vectorization
    DEFAULT_NUM_OUTPUT_COLS = 1024

    # Default size range for n-gram generation
    DEFAULT_MIN_NGRAM_SIZE = 1
    DEFAULT_MAX_NGRAM_SIZE = 2

    _TEMPLATE = "preprocess/text.jinja"

    def __init__(self,
                 var_text_transformers: str,
                 text_cols: Set[str],
                 num_output_cols: int = DEFAULT_NUM_OUTPUT_COLS,
                 min_ngram_size: int = DEFAULT_MIN_NGRAM_SIZE,
                 max_ngram_size: int = DEFAULT_MAX_NGRAM_SIZE,
                 name_prefix: str = "tp"):
        """
        :param var_text_transformers: var name of text_transformers
        :param text_cols: columns to apply text featurization
        :param num_output_cols: number of output cols for each text feature
        :param min_ngram_size: the minimum size for n-gram generation
        :param max_ngram_size: the maximum size for n-gram generation
        """
        super(TextPreprocessor, self).__init__(
            name_prefix, transformer_output_names=[var_text_transformers])

        if not text_cols:
            raise InvalidSectionInputError("text_cols is empty or None")
        if min_ngram_size > max_ngram_size:
            raise InvalidSectionInputError("min_ngram_size cannot be greater than max_ngram_size")
        if min_ngram_size <= 0 or max_ngram_size <= 0:
            raise InvalidSectionInputError("min_ngram_size and max_ngram_size must be positive")

        self._text_cols = text_cols
        self._num_output_cols = num_output_cols
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._var_text_transformers = var_text_transformers

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        return self.template_manager.render_multicells(
            self._TEMPLATE,
            text_cols=self._text_cols,
            var_text_transformers=self._var_text_transformers,
            num_output_cols=self._num_output_cols,
            min_ngram_size=self._min_ngram_size,
            max_ngram_size=self._max_ngram_size)
