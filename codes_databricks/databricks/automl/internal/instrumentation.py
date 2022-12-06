import json
import logging
import uuid
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Tuple, Any

import mlflow
import pandas as pd
import pyspark.pandas as ps
import wrapt
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession

from databricks.automl.shared.errors import AutomlError
from databricks.automl.shared.tags import Tag
from databricks.automl.internal.classification.stats import ClassificationInputStats
from databricks.automl.internal.common.const import DatasetFormat
from databricks.automl.internal.preprocess import SupervisedLearnerDataPreprocessResults
from databricks.automl.internal.stats import PostSamplingStats, InputStats

_logger = logging.getLogger(__name__)


@dataclass
class InputParams:
    dataset_type: str
    target_col_type: str
    data_dir_present: bool
    metric: Optional[str] = None
    max_trials: Optional[int] = None
    timeout_minutes: Optional[int] = None
    experiment_id: Optional[str] = None
    time_col_type: Optional[str] = None
    horizon: Optional[int] = None
    frequency: Optional[str] = None
    output_database_present: Optional[bool] = False
    pos_label_present: Optional[bool] = False
    num_feature_store_lookups: Optional[int] = 0


@dataclass
class Start:
    problem_type: str
    source: str
    input_params: InputParams
    selected_frameworks: List[str]
    imputers: Dict[str, Dict[str, int]]
    feature_schema: Dict[str, int]
    runtime_version: str
    library_versions: Dict[str, str]
    identity_col_schema: Optional[Dict[str, int]] = None
    output_database_present: bool = False


@dataclass
class InputDataStats:
    num_rows: int
    num_cols: int
    num_string_cols: Optional[int]  # Logged as None for forecasting
    num_target_nulls: int  # Number of rows where target is NULL and hence invalid
    class_counts: Optional[List[int]]  # Number of rows in each label (classification only)


@dataclass
class DataStats:
    notebook_id: int
    num_rows: int
    num_cols: int
    precise_sample_fraction: float = 1.0  # sample fraction used in the precise sampling step
    sample_fraction: float = 1.0  # final sample fraction combines both rough and precise sampling steps
    size_mb: Optional[float] = None
    num_string_columns_low_cardinality: Optional[int] = None
    num_string_columns_high_cardinality: Optional[int] = None
    num_string_columns_unique_values: Optional[int] = None
    num_unsupported_cols: Optional[int] = None
    num_constant_cols: Optional[int] = None
    num_missing_value_cols: Optional[int] = None
    num_classes: Optional[int] = None
    num_exclude_cols: int = None  # This was previously named num_exclude_columns before 10.5
    num_array_columns: Optional[int] = None
    sparse_or_dense: Optional[str] = None
    mem_req_data_load_mb: Optional[float] = None
    mem_req_training_mb_dense: Optional[float] = None
    mem_req_training_mb_sparse: Optional[float] = None
    is_imbalanced: Optional[
        bool] = None  # Whether the dataset is imbalanced and using appropriate metrics
    target_label_ratios: Optional[List[
        float]] = None  # Ratios between the size of each class and the largest class (classification only)


@dataclass
class Success:
    experiment_id: str
    num_trials: int
    notebook_ids: List[int]
    best_trial_notebook_id: int
    best_trial_score: float
    is_early_stopped: bool
    output_table_present: bool


@dataclass
class Failure:
    error: str


@wrapt.decorator
def instrumented(func, self, args, kwargs):
    """
    This decorator works with supervised learner and wraps it to instrument the inputs and
    outputs from the function. We log the input parameters, output results and output exceptions if any

    This wrapper generates a run_id that's used to track various events from a given AutoML run. This run_id
    is also passed into the function as kwargs so that the function can use the same run_id to log more events
    about the execution. Eg: logging the data stats event

    :param func: The function to wrap, in this case SupervisedLearner.fit(..)
    :param self: The reference to the instance of the class whose function is wrapped i.e. SupervisedLearner
    :param args: The arguments passed to the function
    :param kwargs: The keyword arguments passed to the function
    :return: Wrapped function execution with instrumentation
    """
    dataset = kwargs.get("dataset", None)
    # dataset might be passed as unnamed args
    if dataset is None:
        dataset = args[0]

    target_col = kwargs["target_col"]
    data_dir = kwargs.get("data_dir", None)
    max_trials = kwargs.get("max_trials", None)
    timeout_minutes = kwargs.get("timeout_minutes", None)
    experiment = kwargs.get("experiment", None)
    time_col = kwargs.get("time_col", None)
    identity_col = kwargs.get("identity_col", None)
    horizon = kwargs.get("horizon", None)
    frequency = kwargs.get("frequency", None)
    output_database = kwargs.get("output_database", None)
    pos_label = kwargs.get("pos_label", None)
    feature_store_lookups = kwargs.get("feature_store_lookups", [])

    imputer_counts = _get_imputer_counts(self, kwargs.get("imputers", {}), dataset)
    metric = _get_metric(self, kwargs.get("metric", None))
    target_col_type = _get_col_type(dataset, target_col)
    time_col_type = _get_col_type(dataset, time_col) if time_col else None

    dataset_type, feature_schema, identity_col_schema = _get_dataset_schema(
        dataset, target_col, time_col, identity_col)

    exclude_frameworks = kwargs.get("exclude_frameworks", [])
    selected_frameworks = _get_selected_frameworks(self, exclude_frameworks)
    experiment_id = _get_experiment_id(experiment)
    source = _get_source(experiment)

    runtime_version = self.spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion",
                                          "unknown")
    library_versions = _get_library_versions()

    # unique ID to track the call to fit(..) which is also passed as an argument to fit(..)
    # to allow re-using the same run-id to log additional events for the same run
    run_id = _generate_run_id()
    kwargs["run_id"] = run_id

    start = Start(
        problem_type=self.problem_type.value,
        source=source,
        input_params=InputParams(
            dataset_type=dataset_type,
            target_col_type=target_col_type,
            data_dir_present=data_dir is not None,
            metric=metric,
            max_trials=max_trials,
            timeout_minutes=timeout_minutes,
            experiment_id=experiment_id,
            time_col_type=time_col_type,
            horizon=horizon,
            frequency=frequency,
            output_database_present=output_database is not None,
            pos_label_present=pos_label is not None,
            num_feature_store_lookups=len(feature_store_lookups) if feature_store_lookups else 0),
        selected_frameworks=selected_frameworks,
        imputers=imputer_counts,
        feature_schema=feature_schema,
        runtime_version=runtime_version,
        library_versions=library_versions,
        identity_col_schema=identity_col_schema)

    _flush_event(spark=self.spark, event=start, run_id=run_id)

    try:
        summary = func(*args, **kwargs)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt) or issubclass(type(e), AutomlError):
            failure = Failure(error=type(e).__name__)
        else:
            failure = Failure(error="unknown")
        _flush_event(spark=self.spark, event=failure, run_id=run_id)
        raise e
    else:
        success = Success(
            experiment_id=summary.experiment.experiment_id,
            num_trials=len(summary.trials),
            notebook_ids=[trial.notebook_id for trial in summary.trials],
            best_trial_notebook_id=summary.best_trial.notebook_id,
            best_trial_score=float(summary.best_trial.evaluation_metric_score),
            is_early_stopped=summary.is_early_stopped,
            output_table_present=summary.output_table_name is not None)
        _flush_event(spark=self.spark, event=success, run_id=run_id)
        return summary


def return_default_on_exception(exception_return_value: Any):
    """
    :param exception_return_value: value that will be returned if the
        decorated function causes an exception
    """

    def decorator(func):
        def try_except_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return exception_return_value

        return try_except_wrapper

    return decorator


@return_default_on_exception("invalid")
def _get_source(experiment: Optional[mlflow.entities.Experiment]) -> str:
    if experiment is None:
        return "api"
    if bool(experiment.tags.get(Tag.SOURCE_GUI, False)):
        return "gui"
    return "api"


@return_default_on_exception("invalid")
def _get_metric(self, metric: Optional[str]) -> str:
    # if metric is None, then we want to log None instead of the default metric
    if metric is None:
        return None
    return self._validate_and_parse_metric(metric).short_name


@return_default_on_exception("invalid")
def _get_experiment_id(experiment: Optional[mlflow.entities.Experiment]) -> str:
    if experiment is None:
        return None
    return experiment.experiment_id


@return_default_on_exception(list())
def _get_selected_frameworks(self, exclude_frameworks: List[str]) -> List[str]:
    exclude_frameworks = self._validate_and_filter_exclude_frameworks(exclude_frameworks)
    return sorted({x.value for x in self.supported_frameworks} - set(exclude_frameworks))


@return_default_on_exception(dict())
def _get_imputer_counts(self, imputers: Dict[str, Union[str, Dict[str, Any]]],
                        dataset: Any) -> Dict[str, Dict[str, int]]:
    validated_imputers = self._validate_and_filter_imputers(imputers, dataset.schema)
    imputer_counts = {}
    for imputer in validated_imputers.values():
        type_ = imputer.type.value
        strategy = imputer.get_sklearn_imputer().strategy
        if type_ in imputer_counts:
            strategy_counts = imputer_counts[type_]
            strategy_counts[strategy] = 1 + strategy_counts.get(strategy, 0)
        else:
            imputer_counts[type_] = {strategy: 1}
    return imputer_counts


@return_default_on_exception((None, None, None))
def _get_dataset_schema(dataset: Any, target_col: str, time_col: Optional[str],
                        identity_col: Optional[Union[str, list]]
                        ) -> Tuple[str, Dict[str, int], Optional[Dict[str, int]]]:
    """
    Parse input dataset and extract data to log
    :param dataset:          Input dataset
    :param identity_col:     Name of time series identity columns in forecasting
    :return: (dataset_type, feature_schema_dictionary, identity_schema_dictionary)
    """
    non_feature_cols = set(filter(lambda x: x is not None, [target_col, time_col]))
    if identity_col is not None:
        if isinstance(identity_col, str):
            non_feature_cols.add(identity_col)
        elif isinstance(identity_col, list):
            non_feature_cols.update(identity_col)

    dataset_type = "invalid"
    feature_types = []
    identity_types = []

    if identity_col is None:
        identity_col = []
    elif isinstance(identity_col, str):
        identity_col = [identity_col]

    if isinstance(dataset, DataFrame):
        dataset_type = DatasetFormat.SPARK.value
        schema_fields = dataset.schema.fields
        feature_types = [
            f.dataType.typeName() for f in schema_fields if f.name not in non_feature_cols
        ]
        identity_types = [f.dataType.typeName() for f in schema_fields if f.name in identity_col]

    elif isinstance(dataset, pd.DataFrame):
        dataset_type = DatasetFormat.PANDAS.value
        dtypes_dict = dataset.dtypes.to_dict()
        feature_types = [
            str(tpe) for col, tpe in dtypes_dict.items() if col not in non_feature_cols
        ]
        identity_types = [str(tpe) for col, tpe in dtypes_dict.items() if col in identity_col]

    elif isinstance(dataset, ps.DataFrame):
        dataset_type = DatasetFormat.PYSPARK_PANDAS.value
        schema_fields = dataset.spark.schema().fields
        feature_types = [
            f.dataType.typeName() for f in schema_fields if f.name not in non_feature_cols
        ]
        identity_types = [f.dataType.typeName() for f in schema_fields if f.name in identity_col]

    feature_counts = dict(Counter(feature_types))
    identity_counts = dict(Counter(identity_types)) if identity_types else None

    return dataset_type, feature_counts, identity_counts


@return_default_on_exception(None)
def _get_col_type(dataset: Any, col_name: str) -> str:
    """
    Get the type of given column name
    :param dataset:  Input Dataset
    :param col_name: Column Name.
    :return: column_type
    """
    col_type = "invalid"

    if isinstance(dataset, DataFrame):
        col_type = str(dataset.schema[col_name].dataType) if col_name in dataset.schema.fieldNames(
        ) else "invalid"

    elif isinstance(dataset, pd.DataFrame):
        col_type = str(dataset.dtypes.to_dict().get(col_name, "invalid"))

    elif isinstance(dataset, ps.DataFrame):
        schema = dataset.spark.schema()
        col_type = str(schema[col_name].dataType) if col_name in schema.fieldNames() else "invalid"

    return col_type


@return_default_on_exception(dict())
def _get_library_versions() -> Dict[str, str]:
    """
    Fetches the versions of the libraries which AutoML depends on
    :return: Map of {library_name: version}
    """
    # TODO(ML-15027): Log only versions that are different from default
    import sklearn
    import xgboost
    import lightgbm
    import pandas
    import mlflow
    import numpy
    import prophet
    from databricks.automl import __version__

    # Starting python 3.8 (See: https://docs.python.org/3.8/whatsnew/3.8.html#changes-in-python-behavior)
    # SyntaxWarning is emitted and SHAP has some code that misuses the identity check
    # TODO: [ML-17981] Remove filterwarnings once SHAP fixes the issue
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="\"is\" with a literal. Did you mean \"==\"?")
        warnings.filterwarnings("ignore", message="\"is not\" with a literal. Did you mean \"!=\"?")

        import shap

    return {
        "automl": __version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "pandas": pandas.__version__,
        "mlflow": mlflow.__version__,
        "shap": shap.__version__,
        "numpy": numpy.__version__,
        "prophet": prophet.__version__,
    }


def _generate_run_id() -> str:
    """
    Wrapper fn to generate run-ID for easier mocking in tests
    :return:
    """
    return str(uuid.uuid4())


def _to_dict(obj):
    """
    Helper function to handle nested objects to dict
    """
    if not hasattr(obj, "__dict__"):
        return obj
    return {key: _to_dict(val) for key, val in obj.__dict__.items()}


def _flush_event(spark, event, run_id) -> None:
    """
    This fn creates a dictionary from the event fields, adds the
    run_id to this dictionary and logs this to the spark instrumentation
    pipeline.

    :param spark: Active SparkSession
    :param event: The event to be logged
    :param run_id: The run_id of the AutoML run
    """
    ev_dict = _to_dict(event)
    ev_dict = {k: v for k, v in ev_dict.items() if v is not None}

    # add session id
    ev_dict["run_id"] = run_id

    blob = json.dumps(ev_dict)
    try:
        spark.sparkContext._gateway.entry_point.getLogger(). \
            logUsage(
            "autoMLRun",
            {"eventType": event.__class__.__name__},
            blob,
        )
    except Exception as e:
        _logger.warning(f"Unable to log instrumentation for AutoML: {e}")


def log_forecast_input_data_stats(spark, num_rows: int, num_cols: int, num_string_cols: int,
                                  num_target_nulls: int, run_id: str) -> None:
    data_stats = InputDataStats(
        num_rows=num_rows,
        num_cols=num_cols,
        num_string_cols=num_string_cols,
        num_target_nulls=num_target_nulls,
        class_counts=None,
    )

    _flush_event(spark=spark, event=data_stats, run_id=run_id)


def log_forecast_data_stats(spark, notebook_id: int, num_rows: int, num_cols: int,
                            run_id: str) -> None:
    """
    :param spark: Active SparkSession
    :param notebook_id: The id of the data exploration notebook
    :param num_rows: Num rows for input dataset
    :param num_cols: Num cols for input dataset
    :param run_id: The run_id of the AutoML run
    """
    data_stats = DataStats(notebook_id=notebook_id, num_rows=num_rows, num_cols=num_cols)

    _flush_event(spark=spark, event=data_stats, run_id=run_id)


def log_classification_input_data_stats(spark: SparkSession, stats: ClassificationInputStats,
                                        run_id: str) -> None:
    input_data_stats = InputDataStats(
        num_rows=stats.num_rows,
        num_cols=stats.num_cols,
        num_string_cols=stats.num_string_cols,
        num_target_nulls=stats.num_target_nulls,
        class_counts=sorted(stats.class_counts.values())
        if stats.class_counts is not None else None)
    _flush_event(spark=spark, event=input_data_stats, run_id=run_id)


def log_supervised_input_data_stats(spark: SparkSession, stats: InputStats, run_id: str) -> None:
    input_data_stats = InputDataStats(
        num_rows=stats.num_rows,
        num_cols=stats.num_cols,
        num_string_cols=stats.num_string_cols,
        num_target_nulls=stats.num_target_nulls,
        class_counts=None)
    _flush_event(spark=spark, event=input_data_stats, run_id=run_id)


def log_supervised_data_stats(spark: SparkSession, notebook_id: int,
                              preprocess_result: SupervisedLearnerDataPreprocessResults,
                              post_sampling_stats: PostSamplingStats, num_exclude_cols: int,
                              num_array_columns: int, dataset_size_bytes: int, is_imbalanced: bool,
                              target_label_ratios: Optional[List[float]], run_id: str) -> None:
    """
    This logs the processed output of the data exploration notebook
    :param spark: Active SparkSession
    :param notebook_id: The id of the data exploration notebook
    :param preprocess_result: Processed output of DataExploration notebook
    :param run_id: The run_id of the AutoML run
    """
    data_stats = DataStats(
        notebook_id=notebook_id,
        num_rows=post_sampling_stats.num_rows,
        num_cols=len(post_sampling_stats.columns),
        precise_sample_fraction=preprocess_result.size_estimator_result.precise_sample_fraction,
        sample_fraction=preprocess_result.size_estimator_result.final_sample_fraction,
        size_mb=(dataset_size_bytes / 10**6),
        num_string_columns_high_cardinality=len(preprocess_result.string_columns_high_cardinality),
        num_string_columns_low_cardinality=len(preprocess_result.string_columns_low_cardinality),
        num_string_columns_unique_values=len(preprocess_result.string_columns_unique_values),
        num_unsupported_cols=len(preprocess_result.unsupported_columns),
        num_constant_cols=len(preprocess_result.constant_columns),
        num_missing_value_cols=len(
            [col for col in post_sampling_stats.columns.values() if col.num_missing > 0]),
        num_classes=preprocess_result.num_classes,
        num_exclude_cols=num_exclude_cols,
        num_array_columns=num_array_columns,
        sparse_or_dense=preprocess_result.size_estimator_result.sparse_or_dense.value,
        mem_req_data_load_mb=preprocess_result.size_estimator_result.mem_req_data_load_mb,
        mem_req_training_mb_dense=preprocess_result.size_estimator_result.mem_req_training_mb_dense,
        mem_req_training_mb_sparse=preprocess_result.size_estimator_result.
        mem_req_training_mb_sparse,
        is_imbalanced=is_imbalanced,
        target_label_ratios=target_label_ratios,
    )
    _flush_event(spark=spark, event=data_stats, run_id=run_id)
