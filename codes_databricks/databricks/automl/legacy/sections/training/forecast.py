from abc import abstractmethod
from typing import List, Optional

import nbformat

from databricks.automl.legacy.problem_type import ProblemType
from databricks.automl.legacy.section import Section
from databricks.automl_runtime.forecast import OFFSET_ALIAS_MAP


class ForecastSection(Section):
    """
    Abstract section class for forecasting models.
    """

    @abstractmethod
    def __init__(self, var_loaded_df: str, var_target_col: str, var_time_col: str, var_id_cols: str,
                 var_frequency_unit: str, var_horizon: str, var_run: str, target_col: str,
                 time_col: str, identity_col: Optional[List[str]], horizon: int,
                 frequency_unit: str, metric: str, timeout: Optional[int],
                 invalid_identities: Optional[List[str]], num_folds: int, experiment_url: str,
                 experiment_id: str, random_state: int, **kwargs):
        self._var_input_df = var_loaded_df
        self._var_target_col = var_target_col
        self._var_time_col = var_time_col
        self._var_id_cols = var_id_cols
        self._var_frequency_unit = var_frequency_unit
        self._var_horizon = var_horizon
        self._var_run = var_run

        self._target_col = target_col
        self._time_col = time_col
        self._identity_col = identity_col
        self._experiment_url = experiment_url
        self._experiment_id = experiment_id
        self._horizon = horizon
        self._frequency_unit = frequency_unit
        self._metric = metric
        self._timeout = timeout
        self._invalid_identities = invalid_identities
        self._num_folds = num_folds
        self._random_state = random_state

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    @abstractmethod
    def name_prefix(self) -> str:
        pass

    @property
    def input_names(self) -> List[str]:
        inputs = [
            self._var_input_df, self._var_time_col,
            self._var_id_cols if self._identity_col is not None else None, self._var_target_col
        ]

        return list(filter(lambda x: x is not None, inputs))

    @property
    def output_names(self) -> List[str]:
        return [self._var_run]

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.FORECAST

    @property
    @abstractmethod
    def cells(self) -> List[nbformat.NotebookNode]:
        pass


class ProphetSection(ForecastSection):
    """
    Section that uses an Prophet to train a model.
    """

    _UNIVARIATE_TRAINING_TEMPLATE = "forecast/univariate_prophet.jinja"
    _MULTIVARIATE_TRAINING_TEMPLATE = "forecast/multivariate_prophet.jinja"

    def __init__(self, var_loaded_df: str, var_target_col: str, var_time_col: str, var_id_cols: str,
                 var_frequency_unit: str, var_horizon: str, var_run: str, target_col: str,
                 time_col: str, identity_col: Optional[List[str]], horizon: int,
                 frequency_unit: str, metric: str, timeout: Optional[int],
                 invalid_identities: Optional[List[str]], num_folds: int, experiment_url: str,
                 experiment_id: str, random_state: int, **kwargs):
        super().__init__(
            var_loaded_df=var_loaded_df,
            var_target_col=var_target_col,
            var_time_col=var_time_col,
            var_id_cols=var_id_cols,
            var_frequency_unit=var_frequency_unit,
            var_horizon=var_horizon,
            var_run=var_run,
            target_col=target_col,
            time_col=time_col,
            identity_col=identity_col,
            horizon=horizon,
            frequency_unit=frequency_unit,
            metric=metric,
            timeout=timeout,
            invalid_identities=invalid_identities,
            num_folds=num_folds,
            experiment_url=experiment_url,
            experiment_id=experiment_id,
            random_state=random_state,
        )
        self._max_evals = kwargs["max_evals"]
        self._country_holidays = kwargs["country_holidays"]
        self._interval_width = kwargs["interval_width"]

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return "fbp"

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        if self._identity_col is None:
            template = self._UNIVARIATE_TRAINING_TEMPLATE
        else:
            template = self._MULTIVARIATE_TRAINING_TEMPLATE

        return self.template_manager.render_multicells(
            template,
            var_input_df=self._var_input_df,
            var_target_col=self._var_target_col,
            var_time_col=self._var_time_col,
            var_id_cols=self._var_id_cols,
            var_frequency_unit=self._var_frequency_unit,
            var_horizon=self._var_horizon,
            var_run=self._var_run,
            target_col=self._target_col,
            time_col=self._time_col,
            identity_col=self._identity_col,
            horizon=self._horizon,
            frequency_unit=self._frequency_unit,
            metric=self._metric,
            timeout=self._timeout,
            invalid_identities=self._invalid_identities,
            num_folds=self._num_folds,
            random_state=self._random_state,
            prefix=self.name_prefix,
            experiment_url=self._experiment_url,
            experiment_id=self._experiment_id,
            max_evals=self._max_evals,
            interval_width=self._interval_width,
            country_holidays=self._country_holidays)


class ArimaSection(ForecastSection):
    """
    Section that uses an ARIMA model to train a model.
    """

    _UNIVARIATE_TRAINING_TEMPLATE = "forecast/univariate_arima.jinja"
    _MULTIVARIATE_TRAINING_TEMPLATE = "forecast/multivariate_arima.jinja"

    def __init__(self, var_loaded_df: str, var_target_col: str, var_time_col: str, var_id_cols: str,
                 var_frequency_unit: str, var_horizon: str, var_run: str, target_col: str,
                 time_col: str, identity_col: Optional[List[str]], horizon: int,
                 frequency_unit: str, metric: str, timeout: Optional[int],
                 invalid_identities: Optional[List[str]], num_folds: int, experiment_url: str,
                 experiment_id: str, random_state: int, **kwargs):
        super().__init__(
            var_loaded_df=var_loaded_df,
            var_target_col=var_target_col,
            var_time_col=var_time_col,
            var_id_cols=var_id_cols,
            var_frequency_unit=var_frequency_unit,
            var_horizon=var_horizon,
            var_run=var_run,
            target_col=target_col,
            time_col=time_col,
            identity_col=identity_col,
            horizon=horizon,
            frequency_unit=frequency_unit,
            metric=metric,
            timeout=timeout,
            invalid_identities=invalid_identities,
            num_folds=num_folds,
            experiment_url=experiment_url,
            experiment_id=experiment_id,
            random_state=random_state,
        )

    @property
    def version(self) -> str:
        return "v1"

    @property
    def name_prefix(self) -> str:
        return "arima"

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        if self._identity_col is None:
            template = self._UNIVARIATE_TRAINING_TEMPLATE
        else:
            template = self._MULTIVARIATE_TRAINING_TEMPLATE

        return self.template_manager.render_multicells(
            template,
            var_input_df=self._var_input_df,
            var_target_col=self._var_target_col,
            var_time_col=self._var_time_col,
            var_id_cols=self._var_id_cols,
            var_frequency_unit=self._var_frequency_unit,
            var_horizon=self._var_horizon,
            var_run=self._var_run,
            target_col=self._target_col,
            time_col=self._time_col,
            identity_col=self._identity_col,
            horizon=self._horizon,
            frequency_unit=self._frequency_unit,
            metric=self._metric,
            timeout=self._timeout,
            invalid_identities=self._invalid_identities,
            num_folds=self._num_folds,
            random_state=self._random_state,
            prefix=self.name_prefix,
            experiment_url=self._experiment_url,
            experiment_id=self._experiment_id,
            offset_alias_map=OFFSET_ALIAS_MAP,
        )
