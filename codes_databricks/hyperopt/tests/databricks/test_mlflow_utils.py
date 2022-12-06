import logging
import os
import unittest

import mlflow
from mlflow.tracking import MlflowClient

from hyperopt import anneal, hp, mlflow_utils, pyll
from hyperopt import SparkTrials
from hyperopt.base import (
    JOB_STATE_DONE,
    JOB_STATE_ERROR,
    JOB_STATE_CANCEL,
    JOB_STATE_NEW,
    STATUS_OK,
    STATUS_FAIL,
    STATUS_NEW,
)
from hyperopt.base import spec_from_misc
from hyperopt.mlflow_utils import _MLflowLogging, _MLflowCompat
from hyperopt.utils import _get_random_id

from hyperopt.tests.test_base import create_fake_trial
from hyperopt.tests.integration.test_spark import (
    BaseSparkContext,
    TestTempDir,
    patch_logger,
)


def _space_to_str(space):
    return _MLflowLogging._format_space_str(str(pyll.as_apply(space)))


class MLflowTestMixin(TestTempDir):
    @classmethod
    def set_up_class_for_mlflow(cls, temp_dir="/tmp"):
        cls.make_tempdir(dir=temp_dir)
        cls.mlflow_uri = "file:" + os.path.join(cls.tempdir, "mlflow")
        mlflow.set_tracking_uri(cls.mlflow_uri)
        logging.info(
            "{test} logging to MLflow URI: {uri}".format(
                test=cls.__name__, uri=cls.mlflow_uri
            )
        )

    @classmethod
    def tear_down_class_for_mlflow(cls):
        cls.remove_tempdir()

    def set_up_for_mlflow(self):
        while mlflow.active_run() is not None:
            mlflow.end_run()
        experiment_name = self.__class__.__name__ + _get_random_id()
        self.experiment_id = mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

    def tear_down_for_mlflow(self):
        if "MLFLOW_RUN_ID" in os.environ:
            del os.environ["MLFLOW_RUN_ID"]
        while mlflow.active_run() is not None:
            mlflow.end_run()
        MlflowClient().delete_experiment(self.experiment_id)


class MLflowUtilsTest(unittest.TestCase, BaseSparkContext, MLflowTestMixin):
    @classmethod
    def setUpClass(cls):
        cls.setup_spark()
        cls.set_up_class_for_mlflow()

    @classmethod
    def tearDownClass(cls):
        cls.tear_down_class_for_mlflow()
        cls.teardown_spark()

    def setUp(self):
        self.set_up_for_mlflow()
        self.spark.conf.set(_MLflowLogging._MLFLOW_INTEGRATION_FEATURE_FLAG, "true")

    def tearDown(self):
        self.tear_down_for_mlflow()

    def _check_metrics(self, method_being_checked, expected_metrics, run):
        actual_metrics = _MLflowCompat.to_dict(run.data.metrics)
        self.assertEqual(
            actual_metrics,
            expected_metrics,
            "{method} failed to log expected metrics.\n"
            "Expected: {expected}\n"
            "Actual: {actual}\n".format(
                method=method_being_checked,
                expected=expected_metrics,
                actual=actual_metrics,
            ),
        )

    def _check_params(self, method_being_checked, expected_params, run):
        actual_params = _MLflowCompat.to_dict(run.data.params)
        self.assertEqual(
            actual_params,
            expected_params,
            "{method} failed to log expected params.\n"
            "Expected: {expected}\n"
            "Actual: {actual}\n".format(
                method=method_being_checked,
                expected=expected_params,
                actual=actual_params,
            ),
        )

    def _check_tags(self, method_being_checked, expected_tags, run):
        actual_tags = _MLflowCompat.to_dict(run.data.tags)
        for t, v in expected_tags.items():
            self.assertIn(
                t,
                actual_tags,
                "{method} failed to log expected tags. "
                "Did not find expected tag {t} in: {actual}.".format(
                    method=method_being_checked, t=t, actual=actual_tags
                ),
            )
            self.assertEqual(
                v,
                actual_tags[t],
                "{method} failed to log expected tags.\n"
                "For tag {t}, expected value {v} but found {a}".format(
                    method=method_being_checked, t=t, v=v, a=actual_tags[t]
                ),
            )

    def _check_space_str(self, actual, expected):
        self.assertEqual(
            actual,
            expected,
            "_format_space_str() returns an incorrect string.\n"
            "Expected: {expected}\n"
            "Actual: {actual}\n".format(expected=expected, actual=actual),
        )

    def test_format_space_str(self):
        orig = "21 uniform\n 22 Literal{0.1}\n  23 Literal{0.5}"

        # Not truncated
        new = _MLflowLogging._format_space_str(orig, len(orig) + 10)
        exp = orig.replace("\n", r"\n")
        self._check_space_str(new, exp)

        # Truncated
        new = _MLflowLogging._format_space_str(orig, len(orig) - 5)
        exp = orig.replace("\n", r"\n")[: len(orig) - 5 - 4] + " ..."
        self._check_space_str(new, exp)

        # Invalid max_len
        new = _MLflowLogging._format_space_str(orig, 1)
        exp = orig.replace("\n", r"\n")
        self._check_space_str(new, exp)

    def test_start_fmin_run(self):
        # Set up SparkTrials
        space = hp.uniform("x", -5, 5)
        algo = anneal.suggest
        max_evals = 4
        max_queue_len = 2
        parallelism = 5
        timeout = 6
        trials = SparkTrials(parallelism=parallelism, timeout=timeout)

        # Test: start_fmin_run
        mlflow_logging = _MLflowLogging()
        with patch_logger("hyperopt-spark") as output:
            parent_run_uuid = mlflow_logging.start_fmin_run(
                trials,
                space=space,
                algo=algo,
                max_evals=max_evals,
                max_queue_len=max_queue_len,
            )
            log_output = output.getvalue().strip()
            self.assertIn(
                _MLflowLogging._HAVE_MLFLOW_MESSAGE,
                log_output,
                "_MLflowLogging.start_fmin_run should have logged message about "
                "detecting MLflow but did not in log: {log_output}".format(
                    log_output=log_output
                ),
            )
        self.assertIsNone(
            mlflow.active_run(), "start_fmin_run should not set active run"
        )
        self.assertEqual(
            os.environ.get("MLFLOW_RUN_ID"),
            parent_run_uuid,
            "start_fmin_run set the incorrect run ID in the MLFLOW_RUN_ID "
            "environment variable",
        )

        # Check for params, tags
        parent_run = MlflowClient().get_run(parent_run_uuid)

        expected_params = {
            "max_evals": str(max_evals),
            "actual_parallelism": str(parallelism),
            "parallelism": str(parallelism),
            "timeout": str(timeout),
            "algo": "hyperopt.anneal",
            "max_queue_len": str(max_queue_len),
        }
        self._check_params("start_fmin_run", expected_params, parent_run)

        expected_tags = {
            "space": _space_to_str(space),
            "runSource": "hyperoptAutoTracking",
        }
        self._check_tags("start_fmin_run", expected_tags, parent_run)

    def test_complete_fmin_run(self):
        # Set up SparkTrials
        trials = SparkTrials(parallelism=2, timeout=1)

        with mlflow.start_run():
            mlflow_logging = _MLflowLogging()
            with patch_logger("hyperopt-spark") as output:
                parent_run_uuid = mlflow_logging.start_fmin_run(
                    trials,
                    space=hp.uniform("x", -5, 5),
                    algo=anneal.suggest,
                    max_evals=4,
                    max_queue_len=2,
                )

                # Update SparkTrials with some fake trials
                trial_templates = [
                    {
                        "tid": 0,
                        "loss": 0.123,
                        "status": STATUS_OK,
                        "state": JOB_STATE_DONE,
                    },
                    {
                        "tid": 1,
                        "loss": None,
                        "status": STATUS_FAIL,
                        "state": JOB_STATE_ERROR,
                    },
                    {
                        "tid": 2,
                        "loss": None,
                        "status": STATUS_FAIL,
                        "state": JOB_STATE_CANCEL,
                    },
                    {
                        "tid": 3,
                        "loss": None,
                        "status": STATUS_FAIL,
                        "state": JOB_STATE_CANCEL,
                    },
                ]
                for template in trial_templates:
                    trials.insert_trial_doc(create_fake_trial(**template))
                trials.refresh()

                # Test: complete_fmin_run
                mlflow_logging.complete_fmin_run(parent_run_uuid, trials)

                log_output = output.getvalue().strip()
                # Verify no exceptions while creating/ending MLflow runs in log output
                self.assertNotIn("Exception", log_output)

            self.assertIsNotNone(
                mlflow.active_run(), "complete_fmin_run should not end the current run"
            )
            self.assertEqual(
                mlflow.active_run().info.run_uuid,
                parent_run_uuid,
                "complete_fmin_run seems to have changed the active run ID",
            )

            parent_run = MlflowClient().get_run(parent_run_uuid)
            expected_metrics = {
                "successful_trials_count": 1,
                "failed_trials_count": 1,
                "cancelled_trials_count": 2,
                "total_trials_count": 4,
                "best_trial_loss": 0.123,
            }
            self._check_metrics("complete_fmin_run", expected_metrics, parent_run)

    def test_back2back_fmin_with_active_run(self):
        # Set up SparkTrials
        space = hp.uniform("x", -5, 5)
        algo = anneal.suggest
        max_evals = 4
        max_queue_len = 2
        parallelism = 5
        timeout = 6

        def run_and_check_fmin(should_append_uuid, expected_params, expected_tags):
            # Test: start_fmin_run
            trials = SparkTrials(parallelism=parallelism, timeout=timeout)
            mlflow_logging = _MLflowLogging()
            parent_run_uuid = mlflow_logging.start_fmin_run(
                trials,
                space=space,
                algo=algo,
                max_evals=max_evals,
                max_queue_len=max_queue_len,
            )
            self.assertIsNotNone(
                mlflow.active_run(), "start_fmin_run failed to set active run"
            )

            # Test: complete_fmin_run
            mlflow_logging.complete_fmin_run(parent_run_uuid, trials)

            self.assertIsNotNone(
                mlflow.active_run(), "complete_fmin_run should not end the current run"
            )

            # Check for params, tags
            parent_run = MlflowClient().get_run(parent_run_uuid)

            params = {
                "max_evals": str(max_evals),
                "actual_parallelism": str(parallelism),
                "parallelism": str(parallelism),
                "timeout": str(timeout),
                "algo": "hyperopt.anneal",
                "max_queue_len": str(max_queue_len),
            }
            if should_append_uuid:
                params = {
                    k + "_" + mlflow_logging._fmin_uuid: v for k, v in params.items()
                }
            expected_params.update(params)
            self._check_params("start_fmin_run", expected_params, parent_run)

            expected_tags.update(
                {
                    "space": _space_to_str(space),
                    "runSource": "hyperoptAutoTracking",
                }
            )
            if should_append_uuid:
                expected_tags[
                    "fmin_uuid_" + mlflow_logging._fmin_uuid
                ] = mlflow_logging._fmin_uuid
            else:
                expected_tags["fmin_uuid"] = mlflow_logging._fmin_uuid
            self._check_tags("start_fmin_run", expected_tags, parent_run)

        with mlflow.start_run():
            exp_params = {}
            exp_tags = {}
            run_and_check_fmin(False, exp_params, exp_tags)
            run_and_check_fmin(True, exp_params, exp_tags)

    def test_back2back_fmin_no_active_run(self):
        # Set up SparkTrials
        space = hp.uniform("x", -5, 5)
        algo = anneal.suggest
        max_evals = 4
        max_queue_len = 2
        parallelism = 5
        timeout = 6

        def run_and_check_fmin():
            # Test: start_fmin_run
            trials = SparkTrials(parallelism=parallelism, timeout=timeout)
            mlflow_logging = _MLflowLogging()
            parent_run_uuid = mlflow_logging.start_fmin_run(
                trials,
                space=space,
                algo=algo,
                max_evals=max_evals,
                max_queue_len=max_queue_len,
            )
            self.assertIsNone(
                mlflow.active_run(), "start_fmin_run should not set active run"
            )
            self.assertIsNotNone(
                os.environ.get("MLFLOW_RUN_ID"),
                "start_fmin_run failed to set active run environment variable",
            )
            self.assertEqual(
                os.environ.get("MLFLOW_RUN_ID"),
                parent_run_uuid,
                "start_fmin_run set the incorrect run ID in the MLFLOW_RUN_ID "
                "environment variable",
            )

            # Test: complete_fmin_run
            mlflow_logging.complete_fmin_run(parent_run_uuid, trials)

            self.assertIsNone(
                mlflow.active_run(),
                "complete_fmin_run should have ended the current run",
            )

            # Check for params, tags
            parent_run = MlflowClient().get_run(parent_run_uuid)

            expected_params = {
                "max_evals": str(max_evals),
                "actual_parallelism": str(parallelism),
                "parallelism": str(parallelism),
                "timeout": str(timeout),
                "algo": "hyperopt.anneal",
                "max_queue_len": str(max_queue_len),
            }
            self._check_params("start_fmin_run", expected_params, parent_run)

            expected_tags = {
                "runSource": "hyperoptAutoTracking",
                "space": _space_to_str(space),
                "fmin_uuid": mlflow_logging._fmin_uuid,
            }
            self._check_tags("start_fmin_run", expected_tags, parent_run)

        run_and_check_fmin()
        run_and_check_fmin()

    def test_trial_run(self):
        parent_run_uuid = mlflow.start_run().info.run_uuid
        mlflow_logging = _MLflowLogging()

        trial = create_fake_trial(tid=1, status=STATUS_NEW, state=JOB_STATE_NEW)
        trial_params = spec_from_misc(trial["misc"])

        # Generates the fmin UUID
        mlflow_logging._update_fmin_uuid()

        # Check start_trial_run
        trial_run_uuid = mlflow_logging.start_trial_run(
            parent_run_uuid, params=trial_params
        )

        self.assertEqual(
            mlflow.active_run().info.run_uuid,
            parent_run_uuid,
            "start_trial_run should not set the active run",
        )
        trial_run = MlflowClient().get_run(trial_run_uuid)

        expected_params = {  # from create_fake_trial
            "z": "1",
        }
        self._check_params("start_trial_run", expected_params, trial_run)

        expected_tags = {
            "mlflow.parentRunId": parent_run_uuid,
            "runSource": "hyperoptAutoTracking",
            "fmin_uuid": mlflow_logging._fmin_uuid,
        }
        self._check_tags("start_trial_run", expected_tags, trial_run)

        # Check complete_trial_run
        mlflow_logging.complete_trial_run(trial_run_uuid, status="success", loss=0.321)

        trial_run = MlflowClient().get_run(trial_run_uuid)
        expected_tags["trial_status"] = "success"
        self._check_tags("complete_trial_run", expected_tags, trial_run)
        expected_metrics = {
            "loss": 0.321,
        }
        self._check_metrics("complete_trial_run", expected_metrics, trial_run)

    def test_mlflow_unavailable(self):
        # This test monkey-patches mlflow_utils to "remove" the MLflow library.
        _mlflow = mlflow_utils._mlflow
        _MlflowClient = mlflow_utils._MlflowClient
        _get_experiment_id = mlflow_utils._get_experiment_id
        _mlflow_entities = mlflow_utils._mlflow_entities
        try:
            mlflow_utils._mlflow = None
            mlflow_utils._MlflowClient = None
            mlflow_utils._get_experiment_id = None
            mlflow_utils._mlflow_entities = None

            trials = SparkTrials(parallelism=2, timeout=1)
            mlflow_logging = _MLflowLogging()
            with patch_logger("hyperopt-spark") as output:
                mlflow_logging.start_fmin_run(
                    trials,
                    space=hp.uniform("x", -5, 5),
                    algo=anneal.suggest,
                    max_evals=4,
                    max_queue_len=2,
                )
                log_output = output.getvalue().strip()
                self.assertIn(
                    _MLflowLogging._NO_MLFLOW_WARNING,
                    log_output,
                    "_MLflowLogging.start_fmin_run should have logged message about "
                    "NOT detecting MLflow but did not in log: {log_output}".format(
                        log_output=log_output
                    ),
                )
            # The follow should pass and not throw exceptions:
            mlflow_logging.complete_fmin_run(0, trials)
            mlflow_logging.start_trial_run(0, None)
            mlflow_logging.complete_trial_run(0, None)
        finally:
            # Reinstate the MLflow library in mlflow_utils
            mlflow_utils._mlflow = _mlflow
            mlflow_utils._MlflowClient = _MlflowClient
            mlflow_utils._get_experiment_id = _get_experiment_id
            mlflow_utils._mlflow_entities = _mlflow_entities

    def _run_method_with_mlflow_error(self, method_name, mlflow_logging, method):
        """
        Helper method for tests which simulate MLflow server errors.
        :param method_name: Name of method being tested, for logging errors
        :param mlflow_logging: Instance of :py:class:`_MLflowLogging` which will be monkey-patched
                               to throw exceptions.
        :param method: Callable to test via `method()`
        """

        def raise_exception(*args, **kwargs):
            raise Exception("random exception")

        mlflow_logging._mlflow_client.log_batch = raise_exception
        mlflow_logging._mlflow_client.log_metric = raise_exception
        true_end_run = mlflow.end_run
        mlflow.end_run = raise_exception

        try:
            with patch_logger("hyperopt-spark") as output:
                method()
                log_output = output.getvalue().strip()
                self.assertTrue(
                    (
                        _MLflowLogging._MLFLOW_LOGGING_FAILED.format(
                            uri=self.mlflow_uri
                        )
                        in log_output
                    )
                    or (
                        _MLflowLogging._MLFLOW_END_RUN_FAILED.format(
                            uri=self.mlflow_uri
                        )
                        in log_output
                    ),
                    "_MLflowLogging.{method} should have logged message about "
                    "MLflow server being temporarily unavailable but did not in log: {log_output}".format(
                        method=method_name, log_output=log_output
                    ),
                )
        finally:
            mlflow.end_run = true_end_run

    def test_mlflow_error_start_fmin_run(self):
        mlflow_logging = _MLflowLogging()
        trials = SparkTrials(parallelism=2, timeout=1)

        def method():
            mlflow_logging.start_fmin_run(
                trials,
                space=hp.uniform("x", -5, 5),
                algo=anneal.suggest,
                max_evals=4,
                max_queue_len=2,
            )

        self._run_method_with_mlflow_error("start_fmin_run", mlflow_logging, method)

    def test_mlflow_error_complete_fmin_run(self):
        mlflow_logging = _MLflowLogging()
        trials = SparkTrials(parallelism=2, timeout=1)
        parent_run_uuid = mlflow_logging.start_fmin_run(
            trials,
            space=hp.uniform("x", -5, 5),
            algo=anneal.suggest,
            max_evals=4,
            max_queue_len=2,
        )
        trials.insert_trial_doc(
            create_fake_trial(tid=0, loss=0.123, status=STATUS_OK, state=JOB_STATE_DONE)
        )
        trials.refresh()

        def method():
            mlflow_logging.complete_fmin_run(parent_run_uuid, trials)

        self._run_method_with_mlflow_error("complete_fmin_run", mlflow_logging, method)

    def test_mlflow_error_start_trial_run(self):
        parent_run_uuid = mlflow.start_run().info.run_uuid
        mlflow_logging = _MLflowLogging()

        trial = create_fake_trial(tid=1, status=STATUS_NEW, state=JOB_STATE_NEW)
        trial_params = spec_from_misc(trial["misc"])

        def method():
            mlflow_logging.start_trial_run(parent_run_uuid, params=trial_params)

        self._run_method_with_mlflow_error("start_trial_run", mlflow_logging, method)

    def test_mlflow_error_complete_trial_run(self):
        parent_run_uuid = mlflow.start_run().info.run_uuid
        mlflow_logging = _MLflowLogging()

        trial = create_fake_trial(tid=1, status=STATUS_NEW, state=JOB_STATE_NEW)
        trial_params = spec_from_misc(trial["misc"])
        trial_run_uuid = mlflow_logging.start_trial_run(
            parent_run_uuid, params=trial_params
        )

        def method():
            mlflow_logging.complete_trial_run(
                trial_run_uuid, status="success", loss=0.321
            )

        self._run_method_with_mlflow_error("complete_trial_run", mlflow_logging, method)

    def test_feature_flag(self):
        mlflow_logging = _MLflowLogging()
        self.assertTrue(
            mlflow_logging._feature_flag_enabled,
            "Expected MLflow tracking to be enabled in this test suite",
        )
        try:
            self.spark.conf.set(
                _MLflowLogging._MLFLOW_INTEGRATION_FEATURE_FLAG, "false"
            )
            mlflow_logging = _MLflowLogging()
            self.assertFalse(
                mlflow_logging._feature_flag_enabled,
                "Expected MLflow tracking to be disabled by feature flag",
            )
        finally:
            self.spark.conf.set(_MLflowLogging._MLFLOW_INTEGRATION_FEATURE_FLAG, "true")
