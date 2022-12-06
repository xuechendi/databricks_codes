import importlib
import os
import time
import uuid

from pkg_resources import parse_version
from pyspark.sql import SparkSession

try:
    _mlflow = importlib.import_module("mlflow")
    _MlflowClient = getattr(
        importlib.import_module("mlflow.tracking.client"), "MlflowClient"
    )
    _get_experiment_id = getattr(
        importlib.import_module("mlflow.tracking.fluent"), "_get_experiment_id"
    )
    _mlflow_entities = importlib.import_module("mlflow.entities")
except ImportError:
    _mlflow = None
    _MlflowClient = None
    _get_experiment_id = None
    _mlflow_entities = None

from hyperopt.exceptions import AllTrialsFailed
from hyperopt.utils import _get_logger

from hyperopt import pyll


logger = _get_logger("hyperopt-spark")

# TODO: CHECK Whether MLflow integration is enabled


class _MLflowCompat:
    """
    This class provides backward compatibility for MLflow
    """

    @staticmethod
    def createMetric(key, value, timestamp):
        """
        Passes the correct arguments when creating Metric objects
        """
        # Note that parse_version() only supports version strings conforming to PEP 440.
        # For example, version strings like "x.x.x-SNAPSHOT" are not supported.
        if parse_version(_mlflow.__version__) > parse_version("0.9.1"):
            return _mlflow_entities.Metric(key, value, timestamp, step=0)
        else:
            return _mlflow_entities.Metric(key, value, timestamp)

    @staticmethod
    def to_dict(x):
        """
        MLflow master changes some objects from list to dict. This function allows
        accessing both types uniformly.
        """
        if isinstance(x, dict):
            return x
        # All these objects have a key and a value
        return {i.key: i.value for i in x}


class _MLflowLogging:
    """
    This class facilitates logging trial-related information into MLflow.

    When the MLflow library is not available, methods will return without doing anything;
    :py:func:`start_fmin_run` will log a warning.
    When the MLflow server is temporarily unreachable or an unknown exception is thrown,
    methods will log warnings.
    """

    # Databricks internal limit
    _MAX_MLFLOW_TAG_LEN = 64 * 1024 - 1

    # Allows turning off worker user logging for testing
    _EN_WORKER_USER_LOGGING = True

    _NO_MLFLOW_WARNING = (
        "Hyperopt cannot find library mlflow. To enable mlflow logging, install the 'mlflow' "
        "library from PyPi."
    )

    _HAVE_MLFLOW_MESSAGE = (
        "Hyperopt with SparkTrials will automatically track trials in MLflow. "
        "To view the MLflow experiment associated with the notebook, click the "
        "'Runs' icon in the notebook context bar on the upper right. There, you "
        "can view all runs."
    )

    _MLFLOW_UNREACHABLE_WARNING = (
        "Hyperopt could not reach the MLflow server at tracking URI: {uri}"
    )

    _MLFLOW_LOGGING_FAILED = (
        "Hyperopt failed to log to MLflow server at tracking URI: {uri}"
    )

    _MLFLOW_END_RUN_FAILED = (
        "Hyperopt failed to execute mlflow.end_run() with tracking URI: {uri}"
    )

    _MLFLOW_INTEGRATION_FEATURE_FLAG = "spark.databricks.mlflow.trackHyperopt.enabled"

    _MLFLOW_INTEGRATION_FEATURE_FLAG_OFF_MESSAGE = (
        "Hyperopt + MLflow integration is feature-flagged off.  To enable automatic tracking in "
        "MLflow, set via: `spark.conf.set('{flag}', 'true')` where `spark` is your SparkSession.".format(
            flag=_MLFLOW_INTEGRATION_FEATURE_FLAG
        )
    )

    def __init__(self):
        if _mlflow is None:
            return
        try:
            self._mlflow_client = _MlflowClient()
            self._tracking_uri = _mlflow.tracking.get_tracking_uri()
        except Exception as e:
            logger.warn(
                "Hyperopt MLflow logging was unable to construct an MLflowClient.\n"
                "Exception: {e}".format(e=e)
            )
            return

        try:
            # canary query to server (which returns None if the experiment does not exist)
            self._mlflow_client.get_experiment_by_name("blah")
        except Exception as e:
            logger.warn(
                _MLflowLogging._MLFLOW_UNREACHABLE_WARNING.format(
                    uri=self._tracking_uri
                )
                + f"\nException: {e}"
            )

        # ID of run created to log fmin results under. May remain None if an active run already
        # exists at the time fmin() is called, in which case we simply log to the user-created
        # active run rather than creating a new run.
        self._created_run_id = None
        # UUID to be appended to parameters, since MLflow forbids overwriting them
        self._fmin_uuid = None
        # Should append fmin UUID
        self._should_append_fmin_uuid = False

        spark = SparkSession.builder.getOrCreate()
        feature_flag_value = spark.conf.get(
            _MLflowLogging._MLFLOW_INTEGRATION_FEATURE_FLAG, "true"
        )
        self._feature_flag_enabled = feature_flag_value.lower() == "true"

    @staticmethod
    def _format_space_str(s, max_len=_MAX_MLFLOW_TAG_LEN):
        """
        Formats the string to be logged for a search space

        :param s: The string to be formatted
        :param max_len: The max length allow for the new formatted string (not s).
                        When max_len is exceeded, the formatted string is truncated to
                        max_len - 4 and a suffix is appended.
        :return: The new formatted string
        """
        # If you change len(suffix), also change "max_len - 4" in he comment above
        suffix = " ..."
        # We replace newlines with spaces since newlines seem to break MLflow logging
        s = s.replace("\n", r"\n")
        if max_len > len(suffix) and len(s) > max_len:
            s = s[: max_len - len(suffix)] + suffix
        return s

    def _update_fmin_uuid(self):
        """
        Generates a UUID for a single fmin() call. Returns None if the tag
        'fmin_uuid" does not exist in the parent run.
        """
        self._fmin_uuid = str(uuid.uuid4())[:6]

    def _append_fmin_uuid(self, objs, obj_type):
        """
        Appends an fmin UUID to the key of a list of objects .

        :param objs: the list of objects to be updated
        :param obj_type: the type of the objects
        :return: the updated list
        """
        return [obj_type(obj.key + "_" + self._fmin_uuid, obj.value) for obj in objs]

    def start_fmin_run(self, spark_trials, space, algo, max_evals, max_queue_len):
        """
        Start an MLflow run for an fmin() call.

        If a run is active, this logs under that run.  Otherwise, this creates a new run and
        sets it as active.  This does NOT end the run.
        If the MLflow library is unavailable, this logs a warning but should not throw an exception.

        :param spark_trials: SparkTrials object whose parameters are logged to the run.
        :return: MLflow run UUID, or None if MLflow is not available
        """
        if _mlflow is None:
            logger.warn(_MLflowLogging._NO_MLFLOW_WARNING)
            return None
        if not self._feature_flag_enabled:
            logger.warn(_MLflowLogging._MLFLOW_INTEGRATION_FEATURE_FLAG_OFF_MESSAGE)
            # By returning None, we make sure that all following calls to logging methods will
            # do nothing.
            return None
        logger.info(_MLflowLogging._HAVE_MLFLOW_MESSAGE)
        self._update_fmin_uuid()

        space_str = _MLflowLogging._format_space_str(str(pyll.as_apply(space)))
        params_to_log = {
            "max_evals": max_evals,
            "actual_parallelism": spark_trials.parallelism,
            "parallelism": spark_trials.user_specified_parallelism,
            "timeout": spark_trials.timeout,
            "algo": algo.__module__,
            "max_queue_len": max_queue_len,
        }
        param_objects = list(
            [_mlflow_entities.Param(k, str(v)) for k, v in params_to_log.items()]
        )
        tag_objects = [
            _mlflow_entities.RunTag("runSource", "hyperoptAutoTracking"),
            _mlflow_entities.RunTag("space", space_str),
        ]
        uuid_tag = [_mlflow_entities.RunTag("fmin_uuid", self._fmin_uuid)]

        try:
            active_run = _mlflow.active_run()
            if active_run is None:
                # Start & end a run in succession. This is equivalent to calling the
                # non-side-effecting MlflowClient.create_run() API, but we use the fluent
                # mlflow.start_run() and mlflow.end_run() APIs to ensure appropriate
                # context-specific tags are set on the run (e.g. tags specifying the URL of the
                # notebook that's currently executing, if running in a Databricks notebook).
                # TODO: once MLflow exposes an API for fetching context tags, we can replace the
                # calls to start_run() and end_run() with a single call to MlflowClient.create_run()
                active_run = _mlflow.start_run()
                self._created_run_id = active_run.info.run_id
                # Pass a status of RUNNING to end_run explicitly so that we set the run's status
                # to RUNNING rather than FINISHED, as we plan to mark the run as terminated after
                # the fmin run completes. Calling end_run here therefore primarily serves to
                # clear MLflow's current "active" run
                _mlflow.end_run(
                    status=_mlflow_entities.RunStatus.to_string(
                        _mlflow_entities.RunStatus.RUNNING
                    )
                )
                os.environ["MLFLOW_RUN_ID"] = self._created_run_id
            else:
                self._created_run_id = None

            run_uuid = active_run.info.run_uuid
            tags = self._mlflow_client.get_run(run_uuid).data.tags
            self._should_append_fmin_uuid = "fmin_uuid" in _MLflowCompat.to_dict(tags)
            if self._should_append_fmin_uuid:
                param_objects = self._append_fmin_uuid(
                    param_objects, _mlflow_entities.Param
                )
                uuid_tag = self._append_fmin_uuid(uuid_tag, _mlflow_entities.RunTag)
            tag_objects.extend(uuid_tag)

            self._mlflow_client.log_batch(
                run_uuid, metrics=[], params=param_objects, tags=tag_objects
            )
            return run_uuid

        except Exception as e:
            logger.warn(
                _MLflowLogging._MLFLOW_LOGGING_FAILED.format(uri=self._tracking_uri)
                + f"\nException: {e}"
            )
            return None

    def log_best_loss(self, run_uuid, best_loss):
        if _mlflow is None or run_uuid is None:
            return
        self._mlflow_client.log_metric(
            run_id=run_uuid, key="best_trial_loss", value=best_loss
        )

    def complete_fmin_run(self, run_uuid, spark_trials):
        """
        Logs final information for an MLflow run for fmin().
        This does NOT end the run.
        :param run_uuid: MLflow run UUID for fmin().
                         This may be None if MLflow failed during :py:func:`start_fmin_run`.
        """
        if _mlflow is None or run_uuid is None:
            return
        metrics_to_log = {
            "successful_trials_count": spark_trials.count_successful_trials(),
            "failed_trials_count": spark_trials.count_failed_trials(),
            "cancelled_trials_count": spark_trials.count_cancelled_trials(),
            "total_trials_count": spark_trials.count_total_trials(),
        }
        try:
            best_trial = spark_trials.best_trial
            metrics_to_log["best_trial_loss"] = best_trial["result"]["loss"]
        except AllTrialsFailed:
            # Updated OSS now throws an exception when all trials failed.
            pass

        timestamp = int(time.time())
        metric_objects = list(
            [
                _MLflowCompat.createMetric(k, v, timestamp)
                for k, v in metrics_to_log.items()
            ]
        )
        try:
            self._mlflow_client.log_batch(
                run_uuid, metrics=metric_objects, params=[], tags=[]
            )
        except Exception as e:
            logger.warn(
                _MLflowLogging._MLFLOW_LOGGING_FAILED.format(uri=self._tracking_uri)
                + f"\nException: {e}"
            )
        finally:
            self._safe_mlflow_end_run()

    def _safe_mlflow_end_run(self):
        try:
            if self._created_run_id is not None:
                self._mlflow_client.set_terminated(self._created_run_id)
            if (
                os.environ.get("MLFLOW_RUN_ID") == self._created_run_id
                and "MLFLOW_RUN_ID" in os.environ
            ):
                del os.environ["MLFLOW_RUN_ID"]
        except Exception as e:
            logger.warning(
                _MLflowLogging._MLFLOW_END_RUN_FAILED.format(uri=self._tracking_uri)
                + f"\nException: {e}"
            )

    def start_trial_run(self, parent_run_uuid, params):
        """
        Start an MLflow run for a single trial.
        This uses the client API and does not change the active experiment or run.

        :param parent_run_uuid: Parent MLflow run ID for fmin() call.  This may be None if MLflow
                                failed during :py:func:`start_fmin_run`; in this case, this method
                                does nothing.
        :param params: dict of hyperparameters to log
        :return: MLflow run ID for this trial
        """
        if _mlflow is None or parent_run_uuid is None:
            return
        exp_id_for_run = _get_experiment_id()
        param_objects = list(
            [_mlflow_entities.Param(k, str(v)) for k, v in params.items()]
        )
        try:
            mlflow_client = self._mlflow_client
            trial_run_id = mlflow_client.create_run(
                experiment_id=exp_id_for_run,
                tags={
                    "mlflow.parentRunId": parent_run_uuid,
                    "fmin_uuid": self._fmin_uuid,
                    "runSource": "hyperoptAutoTracking",
                },
            ).info.run_uuid
            mlflow_client.log_batch(
                run_id=trial_run_id, metrics=[], params=param_objects, tags=[]
            )
            return trial_run_id
        except Exception as e:
            logger.warn(
                _MLflowLogging._MLFLOW_LOGGING_FAILED.format(uri=self._tracking_uri)
                + f"\nException: {e}"
            )
            return None

    def complete_trial_run(self, trial_run_uuid, status, loss=None):
        """
        Logs final information for an MLflow run for a trial, and ends the run.
        This uses the client API and does not change the active experiment or run.
        :param trial_run_uuid: MLflow run ID for this trial.  This may be None if MLflow
                               failed during :py:func:`start_trial_run`; in this case, this method
                               does nothing.
        :param status: status string to log as a tag 'trial_status'
        :param loss: If trial_status is 'success', then this will be logged as metric 'loss'
        """
        if _mlflow is None or trial_run_uuid is None:
            return
        try:
            mlflow_client = self._mlflow_client
            mlflow_client.set_tag(
                run_id=trial_run_uuid, key="trial_status", value=status
            )
            if status == "success" and loss is not None:
                mlflow_client.log_metric(run_id=trial_run_uuid, key="loss", value=loss)
            if status == "success":
                mlflow_status = "FINISHED"
            else:
                mlflow_status = "FAILED"
            mlflow_client.set_terminated(run_id=trial_run_uuid, status=mlflow_status)
        except Exception as e:
            logger.warn(
                _MLflowLogging._MLFLOW_LOGGING_FAILED.format(uri=self._tracking_uri)
                + f"\nException: {e}"
            )
