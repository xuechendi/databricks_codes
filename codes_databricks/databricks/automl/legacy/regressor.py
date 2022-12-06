from typing import List, Iterable

from pyspark.sql.types import DataType

from databricks.automl.legacy.planner import TrialPlanner
from databricks.automl.legacy.preprocess import SupervisedLearnerDataPreprocessor
from databricks.automl.legacy.problem_type import ProblemType, Metric
from databricks.automl.legacy.regression_planner import SGDRegressionPlanner, DecisionTreeRegressorPlanner, \
    RandomForestRegressorPlanner, XGBoostRegressorPlanner, LGBMRegressorPlanner
from databricks.automl.legacy.supervised_learner import SupervisedLearner


class Regressor(SupervisedLearner):
    """
    Implementation of databricks.automl.regress().
    """

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.REGRESSION

    @property
    def default_metric(self) -> Metric:
        return Metric.R2_SCORE

    @classmethod
    def supported_metrics(self) -> Iterable[Metric]:
        return [Metric.R2_SCORE, Metric.MSE, Metric.RMSE, Metric.MAE]

    @classmethod
    def _get_supported_target_types(cls) -> List[DataType]:
        return SupervisedLearnerDataPreprocessor.NUMERIC_TYPES

    def _get_planners(self) -> List[TrialPlanner]:
        return [
            SGDRegressionPlanner, DecisionTreeRegressorPlanner, RandomForestRegressorPlanner,
            XGBoostRegressorPlanner, LGBMRegressorPlanner
        ]
