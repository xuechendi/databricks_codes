from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pyspark.sql.types import StringType

from databricks.automl.legacy.const import DatasetFormat, Framework
from databricks.automl.legacy.context import DataSource
from databricks.automl.legacy.imputers import Imputer
from databricks.automl.legacy.plan import Plan
from databricks.automl.legacy.preprocess import SupervisedLearnerDataPreprocessResults
from databricks.automl.legacy.problem_type import ClassificationTargetTypes, ProblemType
from databricks.automl.legacy.section import Section
from databricks.automl.legacy.sections.training.config import GlobalConfiguration
from databricks.automl.legacy.sections.training.eval_plots import ClassificationEvaluationPlots
from databricks.automl.legacy.sections.training.exit import IPykernelExit
from databricks.automl.legacy.sections.training.inference import SklearnInference
from databricks.automl.legacy.sections.training.input import LoadData
from databricks.automl.legacy.sections.training.preprocess import ArrayPreprocessor, BooleanPreprocessor, \
    ColumnSelectorPreprocessor, DatetimePreprocessor, NumericalPreprocessor, PreprocessSetup, \
    PreprocessFinish, CategoricalPreprocessor, TextPreprocessor
from databricks.automl.legacy.sections.training.shap import ShapFeatureImportancePlot
from databricks.automl.legacy.sections.training.split import SplitData
from databricks.automl.legacy.sections.training.split_by_col import SplitDataByCol


class TrialPlanner(ABC):
    """
    Module that is used to generate plan(s) for a given model type
    """

    def __init__(self, random_state: int):
        self._random_state = random_state

    @abstractmethod
    def generate(self, hyperparameters: Dict[str, Any]) -> Plan:
        """
        Generates a plan that can be executed
        :return: plan
        """
        pass

    @property
    @abstractmethod
    def problem_type(self) -> ProblemType:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_class(self) -> Section:
        pass

    @classmethod
    @abstractmethod
    def framework(cls) -> Framework:
        pass

    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        return {"model": cls}

    @staticmethod
    @abstractmethod
    def requires_data_imputation() -> bool:
        pass


class SklearnPlanner(TrialPlanner):
    """
    Planner module for all sklearn models
    """

    def __init__(self,
                 var_target_col: str,
                 var_X_train: str,
                 var_X_val: str,
                 var_X_test: str,
                 var_y_train: str,
                 var_y_val: str,
                 var_y_test: str,
                 var_preprocessor: str,
                 var_run: str,
                 var_model: str,
                 var_pipeline: str,
                 preprocess_result: SupervisedLearnerDataPreprocessResults,
                 data_source: DataSource,
                 target_col: str,
                 imputers: Dict[str, Imputer],
                 experiment_id: str,
                 experiment_url: str,
                 driver_notebook_url: str,
                 cluster_info: Dict[str, str],
                 sample_fraction: Optional[float],
                 random_state: int,
                 time_col: Optional[str] = None,
                 var_time_col: Optional[str] = None,
                 pos_label: Optional[ClassificationTargetTypes] = None,
                 split_col: Optional[str] = None,
                 sample_weight_col: Optional[str] = None):
        """
        :param var_target_col: variable name for target column
        :param var_X_train:    variable name for training data without label
        :param var_X_val:      variable name for validation data without label
        :param var_X_test:     variable name for test data without label
        :param var_y_train:    variable name for training data label
        :param var_y_val:      variable name for validation data label
        :param var_y_test:     variable name for test data label
        :param var_preprocessor: variable name for preprocessor
        :param var_run:        variable name for the MLflow run
        :param var_model:      variable name for the model that is trained
        :param var_pipeline:   variable name for the pipeline without the model
        :param preprocess_result: result from the data exploration notebook
        :param data_source: source of the training data: either an mlflow run or a dbfs path
        :param target_col:     target column for the label
        :param imputers: dictionary where keys are column names and values are imputation strategies
        :param experiment_id: id of MLflow experiment
        :param experiment_url:  url of MLflow experiment
        :param driver_notebook_url: name of master notebook from where automl is called
        :param cluster_info:        dictionary containing cluster metadata
        :param sample_fraction: Optional sampling fraction used to sample the dataset
        :param time_col: Optional time column to split the dataset by time
        :param var_time_col: variable name of time column. Optional, only used when time_col is given
        :param pos_label: Optional positive class for binary classification
        :param split_col: Optional column that specifies the train/val/test split of each sample
        :param sample_weight_col: Optional column that specifies the weight of each sample
        """
        super().__init__(random_state=random_state)

        self._var_target_col = var_target_col
        self._var_time_col = var_time_col
        self._var_X_train = var_X_train
        self._var_X_val = var_X_val
        self._var_X_test = var_X_test
        self._var_y_train = var_y_train
        self._var_y_val = var_y_val
        self._var_y_test = var_y_test
        self._var_preprocessor = var_preprocessor
        self._var_run = var_run
        self._var_model = var_model
        self._var_pipeline = var_pipeline

        self._date_columns = preprocess_result.date_columns
        self._timestamp_columns = preprocess_result.timestamp_columns
        self._boolean_columns = preprocess_result.boolean_columns
        self._numerical_columns = preprocess_result.numerical_columns
        self._one_hot_encoding_columns = preprocess_result.string_columns_low_cardinality | \
                                         preprocess_result.categorical_numerical_columns_low_cardinality
        self._feature_hashing_columns = preprocess_result.string_columns_high_cardinality
        self._text_columns = preprocess_result.text_columns
        self._array_columns = preprocess_result.array_columns
        self._unsupported_columns = preprocess_result.unsupported_columns | \
                                    preprocess_result.string_columns_extreme_cardinality | \
                                    preprocess_result.string_columns_unique_values | \
                                    preprocess_result.constant_columns

        self._multiclass = preprocess_result.multiclass
        self._has_nulls = sum(preprocess_result.num_nulls.values()) > 0

        self._data_source = data_source
        self._target_col = target_col
        self._time_col = time_col
        self._imputers = imputers
        self._target_col_type = preprocess_result.target_col_type
        self._experiment_id = experiment_id
        self._experiment_url = experiment_url
        self._driver_notebook_url = driver_notebook_url
        self._cluster_info = cluster_info
        self._sparse_or_dense = preprocess_result.size_estimator_result.sparse_or_dense
        self._sample_fraction = sample_fraction

        self._pos_label = pos_label

        self._split_col = split_col
        self._sample_weight_col = sample_weight_col

    @classmethod
    def framework(cls) -> Framework:
        return Framework.SKLEARN

    def generate(self, hyperparameters: Dict[str, Any]) -> Plan:
        config_map = {
            self._var_target_col: self._target_col,
        }
        if self._time_col and self._var_time_col:
            config_map[self._var_time_col] = self._time_col
        conf_section = GlobalConfiguration(
            config_map=config_map,
            model_name=self.model_name,
            experiment_url=self._experiment_url,
            notebook_url=self._driver_notebook_url,
            cluster_info=self._cluster_info)

        var_loaded_df = "df_loaded"
        var_array_transformers = "array_transformers"
        var_bool_transformers = "bool_transformers"
        var_categorical_hash_transformers = "categorical_hash_transformers"
        var_categorical_one_hot_transformers = "categorical_one_hot_transformers"
        var_date_transformers = "date_transformers"
        var_datetime_transformers = "datetime_transformers"
        var_numerical_transformers = "numerical_transformers"
        var_text_transformers = "text_transformers"
        var_column_selector = "col_selector"
        var_supported_cols = "supported_cols"

        input_section = LoadData(
            var_dataframe=var_loaded_df,
            data_source=self._data_source,
            load_format=DatasetFormat.PANDAS,
            problem_type=self.problem_type,
            sample_fraction=self._sample_fraction)

        supported_cols = self._date_columns | self._timestamp_columns | self._boolean_columns | \
                         self._numerical_columns | self._one_hot_encoding_columns | \
                         self._feature_hashing_columns | \
                         self._text_columns | self._array_columns

        column_selector_sections = ColumnSelectorPreprocessor(
            var_column_selector=var_column_selector,
            var_supported_cols=var_supported_cols,
            unsupported_cols=self._unsupported_columns,
            supported_cols=supported_cols)

        preprocess_sections = []

        has_datetime_columns = self._date_columns or self._timestamp_columns
        if has_datetime_columns:
            preprocess_section = DatetimePreprocessor(
                date_columns=self._date_columns,
                timestamp_columns=self._timestamp_columns,
                imputers=self._imputers,
                var_date_transformers=var_date_transformers,
                var_datetime_transformers=var_datetime_transformers)
            preprocess_sections.append(preprocess_section)

        if self._boolean_columns:
            preprocess_section = BooleanPreprocessor(
                boolean_cols=self._boolean_columns,
                imputers=self._imputers,
                var_bool_transformers=var_bool_transformers)
            preprocess_sections.append(preprocess_section)

        if self._numerical_columns:
            preprocess_section = NumericalPreprocessor(
                numerical_cols=self._numerical_columns,
                imputers=self._imputers,
                var_numerical_transformers=var_numerical_transformers)
            preprocess_sections.append(preprocess_section)

        if self._one_hot_encoding_columns or self._feature_hashing_columns:
            preprocess_section = CategoricalPreprocessor(
                one_hot_cols=self._one_hot_encoding_columns,
                hash_cols=self._feature_hashing_columns,
                var_categorical_hash_transformers=var_categorical_hash_transformers,
                var_categorical_one_hot_transformers=var_categorical_one_hot_transformers,
                imputers=self._imputers,
            )
            preprocess_sections.append(preprocess_section)

        if self._text_columns:
            preprocess_section = TextPreprocessor(
                text_cols=self._text_columns, var_text_transformers=var_text_transformers)
            preprocess_sections.append(preprocess_section)

        if self._array_columns:
            preprocess_section = ArrayPreprocessor(
                array_cols=self._array_columns, var_array_transformers=var_array_transformers)
            preprocess_sections.append(preprocess_section)

        transformer_lists_used = []
        for preprocess_section in preprocess_sections:
            transformer_lists_used += preprocess_section.transformer_output_names

        if preprocess_sections:
            setup = PreprocessSetup()
            finish = PreprocessFinish(transformer_lists_used, self._var_preprocessor,
                                      self._sparse_or_dense)
            preprocess_sections = [setup] + preprocess_sections + [finish]
        else:
            self._var_preprocessor = None

        if self._split_col:
            split_section = SplitDataByCol(
                var_input_df=var_loaded_df,
                split_col=self._split_col,
                var_target_col=self._var_target_col,
                var_X_train=self._var_X_train,
                var_X_val=self._var_X_val,
                var_X_test=self._var_X_test,
                var_y_train=self._var_y_train,
                var_y_val=self._var_y_val,
                var_y_test=self._var_y_test,
                time_col=self._time_col,
            )
        else:
            split_section = SplitData(
                var_input_df=var_loaded_df,
                var_target_col=self._var_target_col,
                var_X_train=self._var_X_train,
                var_X_val=self._var_X_val,
                var_X_test=self._var_X_test,
                var_y_train=self._var_y_train,
                var_y_val=self._var_y_val,
                var_y_test=self._var_y_test,
                stratify=self.problem_type == ProblemType.CLASSIFICATION,
                random_state=self._random_state,
                time_col=self._time_col,
                var_time_col=self._var_time_col,
            )

        model_class = self.model_class
        train_section = model_class(
            var_X_train=self._var_X_train,
            var_X_val=self._var_X_val,
            var_X_test=self._var_X_test,
            var_y_train=self._var_y_train,
            var_y_val=self._var_y_val,
            var_y_test=self._var_y_test,
            var_column_selector=var_column_selector,
            var_preprocessor=self._var_preprocessor,
            var_model=self._var_model,
            var_pipeline=self._var_pipeline,
            var_run=self._var_run,
            has_datetime_columns=has_datetime_columns,
            experiment_url=self._experiment_url,
            experiment_id=self._experiment_id,
            hyperparameters=hyperparameters,
            random_state=self._random_state,
            pos_label=self._pos_label,
            multiclass=self._multiclass,
            sample_weight_col=self._sample_weight_col)

        sections = [conf_section, input_section, column_selector_sections] + \
                   preprocess_sections + [split_section, train_section]

        # SHAP does not support datetime column type as input
        if not has_datetime_columns:
            # string labels have dtype==object in pd.DataFrame, for which shap requires predict to be predict_proba
            use_predict_proba = self._target_col_type == StringType
            shap_plot_section = ShapFeatureImportancePlot(
                var_X_train=self._var_X_train,
                var_X_val=self._var_X_val,
                var_model=self._var_model,
                use_predict_proba=use_predict_proba,
                has_nulls=self._has_nulls,  # nulls in validation sample must be imputed or SHAP will fail
                problem_type=self.problem_type,
                random_state=self._random_state,
            )

            sections.append(shap_plot_section)

        inference_section = SklearnInference(var_run=self._var_run)
        sections.append(inference_section)

        if self.problem_type == ProblemType.CLASSIFICATION:
            eval_plots_section = ClassificationEvaluationPlots(
                experiment_id=self._experiment_id,
                var_run=self._var_run,
                multiclass=self._multiclass)
            sections.append(eval_plots_section)

        exit_section = IPykernelExit(var_run=self._var_run)
        sections.append(exit_section)

        plan_name = self.model_name.replace(" ", "")

        plan = Plan(name=plan_name, sections=sections)
        return plan
