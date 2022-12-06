import time
import unittest

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from hyperopt import anneal, fmin, hp
from hyperopt import SparkTrials
from hyperopt.mlflow_utils import _MLflowLogging, _MLflowCompat

from hyperopt.tests.integration.test_spark import (
    BaseSparkContext,
    fn_succeed_within_range,
    patch_logger,
)
from hyperopt.tests.databricks.test_mlflow_utils import MLflowTestMixin


def fn_log_param(x):
    """
    Helper function to test logging params to MLflow from user code
    """
    import mlflow

    mlflow.log_param("input_arg", x)
    return x


def fn_log_param_with_start_run(x):
    """
    Helper function to test logging params from user code within an mlflow.start_run() block
    """
    import mlflow
    import os

    with mlflow.start_run():
        mlflow.log_param("input_arg", x)
    return x


class SparkTrialsMLflowTestCase(unittest.TestCase, BaseSparkContext, MLflowTestMixin):
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

    def tearDown(self):
        self.tear_down_for_mlflow()

    def _get_child_runs(self, mlflow_client, parent_uuid):
        """Return all MLflow Runs for the given parent_uuid"""
        all_infos = mlflow_client.list_run_infos(
            experiment_id=self.experiment_id, run_view_type=ViewType.ALL
        )

        def is_child_run(info):
            tags = _MLflowCompat.to_dict(mlflow_client.get_run(info.run_uuid).data.tags)
            key = "mlflow.parentRunId"
            return key in tags and tags[key] == parent_uuid

        child_ids = [r.run_uuid for r in filter(is_child_run, all_infos)]
        child_runs = [mlflow_client.get_run(child_id) for child_id in child_ids]
        return child_runs

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

    def _run_fmin_and_check_tags(
        self,
        fn,
        space,
        max_evals,
        parallelism,
        timeout,
        expected_trial_status,
        spark_trials=None,
    ):
        """
        Run fmin() and assume it will not throw exceptions.  Check:
         - The run structure is in place: parent run, child runs.
         - Check a few tags, but not most since MLflowUtilsTest covers logging in detail.
        :param expected_trial_status: Check the trial_status logged for each trial run in MLflow.
        :return ID of run generated for fmin()
        """
        if spark_trials is None:
            spark_trials = SparkTrials(parallelism=parallelism, timeout=timeout)
        fmin(
            fn=fn,
            space=space,
            algo=anneal.suggest,
            max_evals=max_evals,
            return_argmin=False,
            trials=spark_trials,
        )

        # Check parent run for fmin()
        active_run = mlflow.active_run()
        self.assertIsNotNone(active_run, "fmin() failed to set active run")
        parent_uuid = active_run.info.run_uuid
        mlflow_client = MlflowClient()
        parent_run = mlflow_client.get_run(parent_uuid)
        expected_tags = {"runSource": "hyperoptAutoTracking"}
        self._check_tags("fmin run", expected_tags, parent_run)

        # Check child runs for trials
        child_runs = self._get_child_runs(mlflow_client, parent_uuid)
        self.assertEqual(
            len(child_runs),
            max_evals,
            "Expected {expected} child runs but found {actual}.".format(
                expected=max_evals, actual=len(child_runs)
            ),
        )
        expected_tags["trial_status"] = expected_trial_status
        for child_run in child_runs:
            self._check_tags("trial run", expected_tags, child_run)
        return parent_uuid

    def test_two_fmin_no_active_run(self):
        max_evals = 2
        timeout = None
        parallelism = 2
        space = hp.uniform("x", -2, 2)

        fmin(
            fn=lambda x: x + 1,
            space=space,
            algo=anneal.suggest,
            max_evals=max_evals,
            return_argmin=False,
            trials=SparkTrials(parallelism=parallelism),
        )
        self.assertIsNone(
            mlflow.active_run(), "fmin() failed to end the current active run"
        )
        fmin(
            fn=lambda x: x + 3,
            space=space,
            algo=anneal.suggest,
            max_evals=max_evals,
            return_argmin=False,
            trials=SparkTrials(parallelism=parallelism),
        )
        self.assertIsNone(
            mlflow.active_run(), "fmin() failed to end the current active run"
        )

    def test_two_fmin_with_active_run(self):
        max_evals = 2
        timeout = None
        parallelism = 2
        space = hp.uniform("x", -2, 2)

        with mlflow.start_run():
            fmin(
                fn=lambda x: x + 1,
                space=space,
                algo=anneal.suggest,
                max_evals=max_evals,
                return_argmin=False,
                trials=SparkTrials(parallelism=parallelism),
            )
            self.assertIsNotNone(
                mlflow.active_run(),
                "fmin() should not have ended the current active run",
            )
            fmin(
                fn=lambda x: x + 3,
                space=space,
                algo=anneal.suggest,
                max_evals=max_evals,
                return_argmin=False,
                trials=SparkTrials(parallelism=parallelism),
            )

            active_run = mlflow.active_run()
            self.assertIsNotNone(
                active_run, "fmin() should not have ended the current active run"
            )

            # Check parent run for fmin()
            parent_uuid = active_run.info.run_uuid
            mlflow_client = MlflowClient()
            parent_run = mlflow_client.get_run(parent_uuid)
            expected_tags = {"runSource": "hyperoptAutoTracking"}
            self._check_tags("fmin run", expected_tags, parent_run)

            # Check child runs for trials. Expecting max_evals*2 child runs
            # since fmin() was called twice.
            child_runs = self._get_child_runs(mlflow_client, parent_uuid)
            self.assertEqual(
                len(child_runs),
                max_evals * 2,
                "Expected {expected} child runs but found {actual}.".format(
                    expected=max_evals * 2, actual=len(child_runs)
                ),
            )
            expected_tags["trial_status"] = "success"
            for child_run in child_runs:
                self._check_tags("trial run", expected_tags, child_run)

    def test_fmin_with_exception(self):
        max_evals = 2
        parallelism = 2

        # Here, we intentionally trigger an exception and check the resulting behavior
        with self.assertRaisesRegexp(IndexError, "list index out of range"):
            fmin(
                fn=lambda x: x + 1,
                space=None,  # Triggers an exception
                algo=anneal.suggest,
                max_evals=max_evals,
                return_argmin=False,
                trials=SparkTrials(parallelism=parallelism),
            )

        self.assertIsNone(
            mlflow.active_run(), "fmin() failed to end the current active run"
        )

    def test_fmin_with_mlflow(self):
        """
        Test:
         - fmin() with mlflow runs as expected
         - The run structure is in place: parent run, child runs.
         - Check a few logged values, but not all since MLflowUtilsTest covers them too.
         - Check that user code parallelized by fmin() logs to a child run of the run created
           by fmin()
        """
        max_evals = 4
        timeout = None
        parallelism = 4
        space = hp.uniform("x", -2, 2)

        for fn in [fn_log_param, fn_log_param_with_start_run]:
            with mlflow.start_run(), patch_logger("hyperopt-spark") as output:
                run_id = self._run_fmin_and_check_tags(
                    fn,
                    space,
                    max_evals=max_evals,
                    parallelism=parallelism,
                    timeout=timeout,
                    expected_trial_status="success",
                )
                child_runs = self._get_child_runs(MlflowClient(), run_id)
                for run in child_runs:
                    assert "input_arg" in run.data.params
                log_output = output.getvalue().strip()
                self.assertIn(
                    _MLflowLogging._HAVE_MLFLOW_MESSAGE,
                    log_output,
                    "fmin() should have logged message about "
                    "detecting MLflow but did not in log: {log_output}".format(
                        log_output=log_output
                    ),
                )

    def test_fmin_with_mlflow_with_user_logging(self):
        """
        Test:
         - fmin() with mlflow runs as expected
         - The run structure is in place: parent run, child runs.
         - Tags logged by users are present. Not testing params or metrics since
           they are handled similarly
        """
        max_evals = 2
        timeout = None
        parallelism = 2
        space = hp.uniform("x", -2, 2)
        expected_tags = {"user_name": "test_user"}

        with mlflow.start_run() as active_run:
            parent_uuid = active_run.info.run_uuid
            self._run_fmin_and_check_tags(
                fn_succeed_within_range,
                space,
                max_evals=max_evals,
                parallelism=parallelism,
                timeout=timeout,
                expected_trial_status="success",
            )
            mlflow.set_tag(*list(expected_tags.items())[0])

        mlflow_client = MlflowClient()
        parent_run = mlflow_client.get_run(parent_uuid)
        self._check_tags("user logging after fmin run", expected_tags, parent_run)

    def test_fmin_with_mlflow_with_trial_failures(self):
        """
        Test:
         - fmin() with mlflow runs as expected, even when some runs fail
         - The run structure is in place: parent run, child runs.
         - Check a few logged values, but not all since MLflowUtilsTest covers them too.
        """
        max_evals = 2
        timeout = None
        parallelism = 2
        space = hp.uniform("x", -6, -4)

        with mlflow.start_run():
            self._run_fmin_and_check_tags(
                fn_succeed_within_range,
                space,
                max_evals=max_evals,
                parallelism=parallelism,
                timeout=timeout,
                expected_trial_status="failure",
            )

    def test_fmin_with_mlflow_with_trial_cancellations(self):
        max_evals = 1
        timeout = 1
        parallelism = 1
        space = hp.uniform("x", -1, 1)

        def fn(x):
            time.sleep(timeout + 4)
            return x

        with mlflow.start_run():
            self._run_fmin_and_check_tags(
                fn,
                space,
                max_evals=max_evals,
                parallelism=parallelism,
                timeout=timeout,
                expected_trial_status="cancelled",
            )

    def test_mlflow_best_trial_loss_update(self):
        max_evals = 5
        parallelism = 1
        space = hp.uniform("x", -2, 2)

        with mlflow.start_run() as initial_run:
            run_uuid = initial_run.info.run_uuid
            fmin(
                fn=lambda x: -time.time(),
                space=space,
                algo=anneal.suggest,
                max_evals=max_evals,
                return_argmin=False,
                trials=SparkTrials(parallelism=parallelism),
            )

            # Check multiple best_trial_loss records
            mlflow_client = MlflowClient()
            best_trial_losses = mlflow_client.get_metric_history(
                run_uuid, "best_trial_loss"
            )
            n = len(best_trial_losses)
            self.assertGreater(
                n,
                1,
                "fmin() should have logged more than 1 best_trial_losses, "
                "but found len(best_trial_losses) = {}".format(n),
            )
            # TODO:
            # for i in range(n-1):
            #    self.assertLess(best_trial_losses[i+1].value, best_trial_losses[i].value,
            #                    "fmin() should only log a new best_trial_loss when a lower loss is found"
            #                    "but for i = {}, {} >= {}"
            #                    .format(i, best_trial_losses[i+1].value, best_trial_losses[i].value))

    def test_fmin_with_feature_flag_off(self):
        try:
            self.spark.conf.set(
                _MLflowLogging._MLFLOW_INTEGRATION_FEATURE_FLAG, "false"
            )
            with patch_logger("hyperopt-spark") as output:
                spark_trials = SparkTrials(parallelism=2)
                fmin(
                    fn=fn_succeed_within_range,
                    space=hp.uniform("x", -2, 2),
                    algo=anneal.suggest,
                    max_evals=2,
                    return_argmin=False,
                    trials=spark_trials,
                )

                # Check to make sure there is NO parent run for fmin()
                active_run = mlflow.active_run()
                self.assertIsNone(
                    active_run,
                    "fmin() should not create or set runs when the feature flag is off",
                )
                log_output = output.getvalue().strip()
                self.assertIn(
                    _MLflowLogging._MLFLOW_INTEGRATION_FEATURE_FLAG_OFF_MESSAGE,
                    log_output,
                    "fmin() should have logged message about MLflow integration being "
                    "feature-flagged off, but it did not in log: {log_output}".format(
                        log_output=log_output
                    ),
                )
        finally:
            self.spark.conf.set(_MLflowLogging._MLFLOW_INTEGRATION_FEATURE_FLAG, "true")
