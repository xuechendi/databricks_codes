from abc import abstractmethod
from typing import Any, Dict, Optional, List, Type

from databricks.automl.internal.confs import ForecastConf
from databricks.automl.internal.common.const import DatasetFormat, Framework
from databricks.automl.internal.context import DataSource
from databricks.automl.internal.forecast_preprocess import ForecastDataPreprocessResults
from databricks.automl.internal.plan import Plan
from databricks.automl.internal.planner import TrialPlanner
from databricks.automl.internal.sections.section import Section
from databricks.automl.internal.sections.training.config import ForecastGlobalConfiguration
from databricks.automl.internal.sections.training.exit import NotebookExit
from databricks.automl.internal.sections.training.forecast import ProphetSection, ArimaSection
from databricks.automl.internal.sections.training.input import LoadData
from databricks.automl.shared.const import Metric, MLFlowFlavor, ProblemType


class ForecastPlanner(TrialPlanner):
    """
    Abstract class for forecasting trial planners.
    """

    @abstractmethod
    def __init__(self, var_target_col: str, var_time_col: str, var_id_cols: str, var_horizon: str,
                 var_frequency_unit: str, data_source: DataSource, target_col: str, time_col: str,
                 identity_col: Optional[List[str]], horizon: int, unit: str,
                 preprocess_result: ForecastDataPreprocessResults, metric: str,
                 timeout: Optional[int], experiment_id: str, experiment_url: str,
                 driver_notebook_url: str, cluster_info: Dict[str, str], random_state: int,
                 **kwargs):
        """
        :param var_target_col: variable name for target column
        :param var_time_col: variable name for time column
        :param var_id_cols: variable name for id columns
        :param var_horizon: variable name for horizon
        :param var_frequency_unit: variable name for frequency unit
        :param data_source: source of the training data: either an mlflow run or a dbfs path
        :param target_col: target column for the label
        :param time_col: time column
        :param identity_col: identity column for multivariate forecasting
        :param horizon: the horizon to forecast
        :param unit: the frequency unit for forecasting
        :param preprocess_result: result of preprocessing
        :param metric: the metric to evaluate models
        :param timeout: maximum time for the experiment to run
        :param experiment_id: id of MLflow experiment
        :param experiment_url: url of MLflow experiment
        :param driver_notebook_url: name of master notebook from where automl is called
        :param cluster_info: dictionary containing cluster metadata
        :param random_state: random seed
        """
        super().__init__(random_state=random_state)

        self._var_target_col = var_target_col
        self._var_time_col = var_time_col
        self._var_id_cols = var_id_cols
        self._var_horizon = var_horizon
        self._var_frequency_unit = var_frequency_unit

        self._data_source = data_source
        self._target_col = target_col
        self._time_col = time_col
        self._identity_col = identity_col
        self._horizon = horizon
        self._unit = unit
        self._metric = metric
        self._timeout = timeout
        self._experiment_id = experiment_id
        self._experiment_url = experiment_url
        self._driver_notebook_url = driver_notebook_url
        self._cluster_info = cluster_info

        self._invalid_identities = preprocess_result.invalid_identities

        confs = ForecastConf.get_conf(identity_col)
        self._num_folds = confs.num_folds

    def generate(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Plan:
        """
        Generates a plan that can be executed.
        """
        conf_section = ForecastGlobalConfiguration(
            config_map={
                self._var_target_col: self._target_col,
                self._var_time_col: self._time_col,
                self._var_frequency_unit: self._unit
            },
            identity_col=self._identity_col,
            horizon=self._horizon,
            var_id_cols=self._var_id_cols,
            var_horizon=self._var_horizon,
            model_name=self.model_name,
            experiment_url=self._experiment_url,
            notebook_url=self._driver_notebook_url,
            cluster_info=self._cluster_info)

        var_loaded_df = "df_loaded"
        var_run = "mlflow_run"

        input_section = LoadData(
            var_dataframe=var_loaded_df,
            data_source=self._data_source,
            problem_type=self.problem_type,
            load_format=DatasetFormat.PYSPARK_PANDAS)

        train_section = self.model_class(
            var_loaded_df=var_loaded_df,
            var_target_col=self._var_target_col,
            var_time_col=self._var_time_col,
            var_id_cols=self._var_id_cols,
            var_frequency_unit=self._var_frequency_unit,
            var_horizon=self._var_horizon,
            var_run=var_run,
            target_col=self._target_col,
            time_col=self._time_col,
            identity_col=self._identity_col,
            horizon=self._horizon,
            frequency_unit=self._unit,
            metric=self._metric,
            timeout=self._timeout,
            invalid_identities=self._invalid_identities,
            num_folds=self._num_folds,
            experiment_url=self._experiment_url,
            experiment_id=self._experiment_id,
            random_state=self._random_state,
            **self.additional_properties)

        sections = [conf_section, input_section] + [train_section]
        exit_section = NotebookExit(var_run=var_run)

        sections += [exit_section]

        plan_name = self.model_name.replace(" ", "")

        plan = Plan(name=plan_name, sections=sections)
        return plan

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.FORECAST

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_class(self) -> Type[Section]:
        pass

    @staticmethod
    @abstractmethod
    def requires_data_imputation() -> bool:
        pass

    @staticmethod
    @abstractmethod
    def supports_missing_values() -> bool:
        pass

    @classmethod
    @abstractmethod
    def framework(cls) -> Framework:
        pass

    @classmethod
    @abstractmethod
    def mlflow_flavor(cls) -> MLFlowFlavor:
        pass

    @classmethod
    @abstractmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        pass


class ProphetPlanner(ForecastPlanner):
    """
    Module that is used to generate plan(s) for the Prophet model
    """

    def __init__(self, var_target_col: str, var_time_col: str, var_id_cols: str, var_horizon: str,
                 var_frequency_unit: str, data_source: DataSource, target_col: str, time_col: str,
                 identity_col: Optional[List[str]], horizon: int, unit: str,
                 preprocess_result: ForecastDataPreprocessResults, metric: str,
                 timeout: Optional[int], experiment_id: str, experiment_url: str,
                 driver_notebook_url: str, cluster_info: Dict[str, str], random_state: int,
                 **kwargs):
        super().__init__(
            var_target_col=var_target_col,
            var_time_col=var_time_col,
            var_id_cols=var_id_cols,
            var_horizon=var_horizon,
            var_frequency_unit=var_frequency_unit,
            data_source=data_source,
            target_col=target_col,
            time_col=time_col,
            identity_col=identity_col,
            horizon=horizon,
            unit=unit,
            preprocess_result=preprocess_result,
            metric=metric,
            timeout=timeout,
            experiment_id=experiment_id,
            experiment_url=experiment_url,
            driver_notebook_url=driver_notebook_url,
            cluster_info=cluster_info,
            random_state=random_state)
        self.additional_properties = {
            "max_evals": kwargs["max_evals"],
            "interval_width": kwargs.get("interval_width", 0.95),
            "country_holidays": kwargs.get("country_holidays", "US")
        }

    @property
    def model_name(self) -> str:
        return "Prophet"

    @property
    def model_class(self) -> Type[Section]:
        return ProphetSection

    @staticmethod
    def requires_data_imputation() -> bool:
        return False

    @staticmethod
    def supports_missing_values() -> bool:
        return True

    @classmethod
    def framework(cls) -> Framework:
        return Framework.PROPHET

    @classmethod
    def mlflow_flavor(cls) -> MLFlowFlavor:
        return MLFlowFlavor.PROPHET

    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        return {"interval_width": [0.8, 0.95], "country_holidays": [None, "US"], "model": [cls]}


class ArimaPlanner(ForecastPlanner):
    """
    Module that is used to generate plan(s) for the ARIMA model
    """

    def __init__(self, var_target_col: str, var_time_col: str, var_id_cols: str, var_horizon: str,
                 var_frequency_unit: str, data_source: DataSource, target_col: str, time_col: str,
                 identity_col: Optional[List[str]], horizon: int, unit: str,
                 preprocess_result: ForecastDataPreprocessResults, metric: str,
                 timeout: Optional[int], experiment_id: str, experiment_url: str,
                 driver_notebook_url: str, cluster_info: Dict[str, str], random_state: int,
                 **kwargs):
        super().__init__(
            var_target_col=var_target_col,
            var_time_col=var_time_col,
            var_id_cols=var_id_cols,
            var_horizon=var_horizon,
            var_frequency_unit=var_frequency_unit,
            data_source=data_source,
            target_col=target_col,
            time_col=time_col,
            identity_col=identity_col,
            horizon=horizon,
            unit=unit,
            preprocess_result=preprocess_result,
            metric=metric,
            timeout=timeout,
            experiment_id=experiment_id,
            experiment_url=experiment_url,
            driver_notebook_url=driver_notebook_url,
            cluster_info=cluster_info,
            random_state=random_state)
        self.additional_properties = {}

    @property
    def model_name(self) -> str:
        return "ARIMA"

    @property
    def model_class(self) -> Type[Section]:
        return ArimaSection

    @staticmethod
    def requires_data_imputation() -> bool:
        return True

    @staticmethod
    def supports_missing_values() -> bool:
        return False

    @classmethod
    def framework(cls) -> Framework:
        return Framework.ARIMA

    @classmethod
    def mlflow_flavor(cls) -> MLFlowFlavor:
        return MLFlowFlavor.ARIMA

    @classmethod
    def get_hyperparameter_search_space(cls) -> Dict[str, Any]:
        return {"model": [cls]}
