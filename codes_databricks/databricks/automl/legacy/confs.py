from abc import ABC
from typing import List


class InternalConfs:
    """This class contains internal configuration flags used by AutoML.

    The flags in this class can be used for staging or experimental purpose.
    For example, when we are developing feature `foo`, we can have a `ENABLE_FOO` flag
    default to `False` so that it is not visible to customer, but can be set to `True`
    in unit test or manual test.

    All flags in this file are expected to be removed at some point when the feature
    is completed or abandoned. For each flag, please add a TODO with a corresponding
    jira ticket on when it should be removed."""

    # When set to True, the sparse matrix path will be enabled.
    # TODO(ML-21513) Remove this flag after sparse matrix has been well benchmarked and tested.
    ENABLE_SPARSE_MATRIX = True

    # When set to True, do train/val/test split on driver
    ENABLE_TRAIN_TEST_SPLIT_DRIVER = True


class ForecastConf(ABC):
    """
    Configurations for the Forecast
    """
    DEFAULT_MAX_TRIALS = 5
    DEFAULT_MAX_TRIALS_IN_PARALLEL = 2
    DEFAULT_INITIAL_PERIOD = 100

    def __init__(self,
                 max_evals: int,
                 max_cross_validation_folds: int,
                 max_trials: int = DEFAULT_MAX_TRIALS,
                 num_trials_in_parallel: int = DEFAULT_MAX_TRIALS_IN_PARALLEL,
                 initial_period: int = DEFAULT_INITIAL_PERIOD):
        self._max_trials = max_trials
        self._max_evals = max_evals
        self._max_cross_validation_folds = max_cross_validation_folds
        self._num_trials_in_parallel = num_trials_in_parallel
        self._initial_period = initial_period

    @property
    def max_trials(self):
        return self._max_trials

    @property
    def max_evals(self):
        return self._max_evals

    @property
    def num_folds(self):
        return self._max_cross_validation_folds

    @property
    def num_trials_in_parallel(self):
        return self._num_trials_in_parallel

    @property
    def initial_period(self):
        return self._initial_period

    @classmethod
    def get_conf(cls, identity_cols: List[str]):
        if not identity_cols:
            return UnivariateForecastConf()
        elif len(identity_cols) == 1:
            return MultivariateSingleIdConf()
        return MultivariateMultiIdConf()


class UnivariateForecastConf(ForecastConf):
    def __init__(self):
        super().__init__(max_evals=10, max_cross_validation_folds=20)


class MultivariateSingleIdConf(ForecastConf):
    def __init__(self):
        super().__init__(max_evals=10, max_cross_validation_folds=5)


class MultivariateMultiIdConf(ForecastConf):
    def __init__(self):
        super().__init__(max_evals=1, max_cross_validation_folds=2)
