import contextlib
import concurrent.futures
import logging
import multiprocessing
import os
import shutil
import signal
import tempfile
import time
import timeit
import traceback
import unittest

from six import StringIO

import numpy as np
import pyspark
from pyspark.sql import SparkSession
from six import StringIO

from hyperopt import SparkTrials, anneal, base, fmin, hp, rand
from hyperopt.base import STATUS_OK
from hyperopt.spark import (
    FMIN_CANCELLED_REASON_TIMEOUT,
    FMIN_CANCELLED_REASON_EARLY_STOPPING,
)

from hyperopt.tests.unit.test_fmin import test_quadratic1_tpe
from py4j.clientserver import ClientServer


@contextlib.contextmanager
def patch_logger(name, level=logging.INFO):
    """patch logger and give an output"""
    io_out = StringIO()
    log = logging.getLogger(name)
    log.setLevel(level)
    log.handlers = []
    handler = logging.StreamHandler(io_out)
    log.addHandler(handler)
    try:
        yield io_out
    finally:
        log.removeHandler(handler)


class TestTempDir:
    @classmethod
    def make_tempdir(cls, dir="/tmp"):
        """
        :param dir: Root directory in which to create the temp directory
        """
        cls.tempdir = tempfile.mkdtemp(prefix="hyperopt_tests_", dir=dir)

    @classmethod
    def remove_tempdir(cls):
        shutil.rmtree(cls.tempdir)


class BaseSparkContext:
    """
    Mixin which sets up a SparkContext for tests
    """

    NUM_SPARK_EXECUTORS = 4

    @classmethod
    def setup_spark(cls):
        cls._spark = (
            SparkSession.builder.master(
                f"local[{BaseSparkContext.NUM_SPARK_EXECUTORS}]"
            )
            .appName(cls.__name__)
            .getOrCreate()
        )
        cls._sc = cls._spark.sparkContext
        cls._pin_mode_enabled = isinstance(cls._sc._gateway, ClientServer)
        cls.checkpointDir = tempfile.mkdtemp()
        cls._sc.setCheckpointDir(cls.checkpointDir)
        # Small tests run much faster with spark.sql.shuffle.partitions=4
        cls._spark.conf.set("spark.sql.shuffle.partitions", "4")

    @classmethod
    def teardown_spark(cls):
        cls._spark.stop()
        cls._sc = None
        shutil.rmtree(cls.checkpointDir)

    @property
    def spark(self):
        return self._spark

    @property
    def sc(self):
        return self._sc


class TestSparkContext(unittest.TestCase, BaseSparkContext):
    @classmethod
    def setUpClass(cls):
        cls.setup_spark()

    @classmethod
    def tearDownClass(cls):
        cls.teardown_spark()

    def test_spark_context(self):
        rdd1 = self.sc.parallelize(range(10), 10)
        rdd2 = rdd1.map(lambda x: x + 1)
        sum2 = rdd2.sum()
        assert sum2 == 55


def fn_succeed_within_range(x):
    """
    Test function to test the handling failures for `fmin`. When run `fmin` with `max_evals=8`,
    it has 7 successful trial runs and 1 failed run.
    :param x:
    :return: 1 when -3 < x < 3, and RuntimeError otherwise
    """
    if -3 < x < 3:
        return 1
    else:
        raise RuntimeError


class FMinTestCase(unittest.TestCase, BaseSparkContext):
    @classmethod
    def setUpClass(cls):
        cls.setup_spark()
        cls._sc.setLogLevel("OFF")

    @classmethod
    def tearDownClass(cls):
        cls.teardown_spark()

    def sparkSupportsJobCancelling(self):
        return hasattr(self.sc.parallelize([1]), "collectWithJobGroup")

    def check_run_status(
        self, spark_trials, output, num_total, num_success, num_failure
    ):
        self.assertEqual(
            spark_trials.count_total_trials(),
            num_total,
            "Wrong number of total trial runs: Expected {e} but got {r}.".format(
                e=num_total, r=spark_trials.count_total_trials()
            ),
        )
        self.assertEqual(
            spark_trials.count_successful_trials(),
            num_success,
            "Wrong number of successful trial runs: Expected {e} but got {r}.".format(
                e=num_success, r=spark_trials.count_successful_trials()
            ),
        )
        self.assertEqual(
            spark_trials.count_failed_trials(),
            num_failure,
            "Wrong number of failed trial runs: Expected {e} but got {r}.".format(
                e=num_failure, r=spark_trials.count_failed_trials()
            ),
        )
        log_output = output.getvalue().strip()
        self.assertIn(
            "Total Trials: " + str(num_total),
            log_output,
            """Logging "Total Trials: {num}" missing from the log: {log}""".format(
                num=str(num_total), log=log_output
            ),
        )
        self.assertIn(
            str(num_success) + " succeeded",
            log_output,
            """Logging "{num} succeeded " missing from the log: {log}""".format(
                num=str(num_success), log=log_output
            ),
        )
        self.assertIn(
            str(num_failure) + " failed",
            log_output,
            """ Logging "{num} failed " missing from the log: {log}""".format(
                num=str(num_failure), log=log_output
            ),
        )

        if spark_trials.count_failed_trials() > 0:
            self.assertIn(spark_trials._ERROR_MESSAGE_WORKER_LOGS, log_output)

    def assert_task_succeeded(self, log_output, task):
        self.assertIn(
            f"trial {task} task thread exits normally",
            log_output,
            """Debug info "trial {task} task thread exits normally" missing from log:
             {log_output}""".format(
                task=task, log_output=log_output
            ),
        )

    def assert_task_failed(self, log_output, task):
        self.assertIn(
            f"trial {task} task thread catches an exception",
            log_output,
            """Debug info "trial {task} task thread catches an exception" missing from log:
             {log_output}""".format(
                task=task, log_output=log_output
            ),
        )

    def check_cancellation_flags(self, spark_trials, message):
        self.assertTrue(
            spark_trials._fmin_cancelled,
            "SparkTrials._fmin_cancelled flag should be true",
        )
        self.assertEqual(
            spark_trials._fmin_cancelled_reason,
            message,
            """SparkTrials._fmin_cancelled_reason should be "{m}", but got "{r}"
                         """.format(
                m=message, r=spark_trials._fmin_cancelled_reason
            ),
        )

    def test_quadratic1_tpe(self):
        # TODO: Speed this up or remove it since it is slow (1 minute on laptop)
        spark_trials = SparkTrials(parallelism=4)
        test_quadratic1_tpe(spark_trials)

    def test_trial_run_info(self):
        spark_trials = SparkTrials(parallelism=4)

        with patch_logger("hyperopt-spark") as output:
            fmin(
                fn=fn_succeed_within_range,
                space=hp.uniform("x", -5, 5),
                algo=anneal.suggest,
                max_evals=8,
                return_argmin=False,
                trials=spark_trials,
                rstate=np.random.default_rng(99),
            )
            self.check_run_status(
                spark_trials, output, num_total=8, num_success=6, num_failure=2
            )

        expected_result = {"loss": 1.0, "status": "ok"}
        for trial in spark_trials._dynamic_trials:
            if trial["state"] == base.JOB_STATE_DONE:
                self.assertEqual(
                    trial["result"],
                    expected_result,
                    "Wrong result has been saved: Expected {e} but got {r}.".format(
                        e=expected_result, r=trial["result"]
                    ),
                )
            elif trial["state"] == base.JOB_STATE_ERROR:
                err_message = trial["misc"]["error"][1]
                self.assertIn(
                    "RuntimeError",
                    err_message,
                    "Missing {e} in {r}.".format(e="RuntimeError", r=err_message),
                )
                self.assertIn(
                    "Traceback (most recent call last)",
                    err_message,
                    "Missing {e} in {r}.".format(e="Traceback", r=err_message),
                )

        num_success = spark_trials.count_by_state_unsynced(base.JOB_STATE_DONE)
        self.assertEqual(
            num_success,
            6,
            "Wrong number of successful trial runs: Expected {e} but got {r}.".format(
                e=6, r=num_success
            ),
        )
        num_failure = spark_trials.count_by_state_unsynced(base.JOB_STATE_ERROR)
        self.assertEqual(
            num_failure,
            2,
            "Wrong number of failed trial runs: Expected {e} but got {r}.".format(
                e=2, r=num_failure
            ),
        )

    def test_accepting_sparksession(self):
        spark_trials = SparkTrials(
            parallelism=2, spark_session=SparkSession.builder.getOrCreate()
        )

        fmin(
            fn=lambda x: x + 1,
            space=hp.uniform("x", 5, 8),
            algo=anneal.suggest,
            max_evals=2,
            trials=spark_trials,
        )

    def test_parallelism_arg(self):
        spark_default_parallelism = 2
        default_parallelism = spark_default_parallelism

        # Test requested_parallelism is None or negative values.
        for requested_parallelism in [None, -1]:
            with patch_logger("hyperopt-spark") as output:
                parallelism = SparkTrials._decide_parallelism(
                    requested_parallelism=requested_parallelism,
                    spark_default_parallelism=spark_default_parallelism,
                )
                self.assertEqual(
                    parallelism,
                    default_parallelism,
                    "Failed to set parallelism to be default parallelism ({p})"
                    " ({e})".format(p=parallelism, e=default_parallelism),
                )
                log_output = output.getvalue().strip()
                self.assertIn(
                    "Because the requested parallelism was None or a non-positive value, "
                    "parallelism will be set to ({d})".format(d=default_parallelism),
                    log_output,
                    """set to default parallelism missing from log: {log_output}""".format(
                        log_output=log_output
                    ),
                )

        # Test requested_parallelism exceeds hard cap
        with patch_logger("hyperopt-spark") as output:
            parallelism = SparkTrials._decide_parallelism(
                requested_parallelism=SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED + 1,
                spark_default_parallelism=spark_default_parallelism,
            )
            self.assertEqual(
                parallelism,
                SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED,
                "Failed to limit parallelism ({p}) to MAX_CONCURRENT_JOBS_ALLOWED ({e})".format(
                    p=parallelism, e=SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED
                ),
            )
            log_output = output.getvalue().strip()
            self.assertIn(
                "SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED ({c})".format(
                    c=SparkTrials.MAX_CONCURRENT_JOBS_ALLOWED
                ),
                log_output,
                """MAX_CONCURRENT_JOBS_ALLOWED value missing from log: {log_output}""".format(
                    log_output=log_output
                ),
            )

    def test_all_successful_trials(self):
        spark_trials = SparkTrials(parallelism=1)
        with patch_logger("hyperopt-spark", logging.DEBUG) as output:
            fmin(
                fn=fn_succeed_within_range,
                space=hp.uniform("x", -1, 1),
                algo=anneal.suggest,
                max_evals=1,
                trials=spark_trials,
            )
            log_output = output.getvalue().strip()

            self.assertEqual(spark_trials.count_successful_trials(), 1)
            self.assertIn(
                "fmin thread exits normally",
                log_output,
                """Debug info "fmin thread exits normally" missing from 
                log: {log_output}""".format(
                    log_output=log_output
                ),
            )
            self.assert_task_succeeded(log_output, 0)

    def test_all_failed_trials(self):
        spark_trials = SparkTrials(parallelism=1)
        with patch_logger("hyperopt-spark", logging.DEBUG) as output:
            fmin(
                fn=fn_succeed_within_range,
                space=hp.uniform("x", 5, 10),
                algo=anneal.suggest,
                max_evals=1,
                trials=spark_trials,
                return_argmin=False,
            )
            log_output = output.getvalue().strip()

            self.assertEqual(spark_trials.count_failed_trials(), 1)
            self.assert_task_failed(log_output, 0)

        spark_trials = SparkTrials(parallelism=4)
        # Here return_argmin is True (by default) and an exception should be thrown
        with self.assertRaisesRegexp(Exception, "There are no evaluation tasks"):
            fmin(
                fn=fn_succeed_within_range,
                space=hp.uniform("x", 5, 8),
                algo=anneal.suggest,
                max_evals=2,
                trials=spark_trials,
            )

    def test_timeout_without_job_cancellation(self):
        timeout = 4
        spark_trials = SparkTrials(parallelism=1, timeout=timeout)
        spark_trials._spark_supports_job_cancelling = False

        def fn(x):
            time.sleep(0.5)
            return x

        with patch_logger("hyperopt-spark", logging.DEBUG) as output:
            fmin(
                fn=fn,
                space=hp.uniform("x", -1, 1),
                algo=anneal.suggest,
                max_evals=10,
                trials=spark_trials,
                max_queue_len=1,
                show_progressbar=False,
                return_argmin=False,
            )
            log_output = output.getvalue().strip()

            self.assertTrue(spark_trials._fmin_cancelled)
            self.assertEqual(spark_trials._fmin_cancelled_reason, "fmin run timeout")
            self.assertGreater(spark_trials.count_successful_trials(), 0)
            self.assertGreater(spark_trials.count_cancelled_trials(), 0)
            self.assertIn(
                "fmin is cancelled, so new trials will not be launched",
                log_output,
                """ "fmin is cancelled, so new trials will not be launched" missing from log:
                {log_output}""".format(
                    log_output=log_output
                ),
            )
            self.assertIn(
                "SparkTrials will block",
                log_output,
                """ "SparkTrials will block" missing from log: {log_output}""".format(
                    log_output=log_output
                ),
            )
            self.assert_task_succeeded(log_output, 0)

    def test_timeout_without_job_cancellation_fmin_timeout(self):
        timeout = 5
        spark_trials = SparkTrials(parallelism=1)
        spark_trials._spark_supports_job_cancelling = False

        def fn(x):
            time.sleep(1)
            return x

        with patch_logger("hyperopt-spark", logging.DEBUG) as output:
            fmin(
                fn=fn,
                space=hp.uniform("x", -1, 1),
                algo=anneal.suggest,
                max_evals=10,
                timeout=timeout,
                trials=spark_trials,
                max_queue_len=1,
                show_progressbar=False,
                return_argmin=False,
                rstate=np.random.default_rng(99),
            )
            log_output = output.getvalue().strip()

            self.check_cancellation_flags(spark_trials, FMIN_CANCELLED_REASON_TIMEOUT)
            self.assertGreater(spark_trials.count_successful_trials(), 0)
            self.assertGreater(spark_trials.count_cancelled_trials(), 0)
            self.assertIn(
                "fmin is cancelled, so new trials will not be launched",
                log_output,
                """ "fmin is cancelled, so new trials will not be launched" missing from log:
                {log_output}""".format(
                    log_output=log_output
                ),
            )
            self.assertIn(
                "SparkTrials will block",
                log_output,
                """ "SparkTrials will block" missing from log: {log_output}""".format(
                    log_output=log_output
                ),
            )
            self.assert_task_succeeded(log_output, 0)

    def test_timeout_with_job_cancellation(self):
        if not self.sparkSupportsJobCancelling():
            print(
                "Skipping timeout test since this Apache PySpark version does not "
                "support cancelling jobs by job group ID."
            )
            return

        timeout = 2
        spark_trials = SparkTrials(parallelism=4, timeout=timeout)

        def fn(x):
            if x < 0:
                time.sleep(timeout + 20)
                raise Exception("Task should have been cancelled")
            else:
                time.sleep(1)
            return x

        # Test 1 cancelled trial.  Examine logs.
        with patch_logger("hyperopt-spark", logging.DEBUG) as output:
            fmin(
                fn=fn,
                space=hp.uniform("x", -2, 0),
                algo=anneal.suggest,
                max_evals=1,
                trials=spark_trials,
                max_queue_len=1,
                show_progressbar=False,
                return_argmin=False,
                rstate=np.random.default_rng(4),
            )
            log_output = output.getvalue().strip()

            self.check_cancellation_flags(spark_trials, FMIN_CANCELLED_REASON_TIMEOUT)
            self.assertEqual(spark_trials.count_cancelled_trials(), 1)
            self.assertIn(
                "Cancelling all running jobs",
                log_output,
                """ "Cancelling all running jobs" missing from log: {log_output}""".format(
                    log_output=log_output
                ),
            )
            self.assertIn(
                "trial task 0 cancelled",
                log_output,
                """ "trial task 0 cancelled" missing from log: {log_output}""".format(
                    log_output=log_output
                ),
            )
            self.assert_task_failed(log_output, 0)

        # Test mix of successful and cancelled trials.
        spark_trials = SparkTrials(parallelism=4, timeout=4)
        fmin(
            fn=fn,
            space=hp.uniform("x", -0.25, 5),
            algo=anneal.suggest,
            max_evals=6,
            trials=spark_trials,
            max_queue_len=1,
            show_progressbar=False,
            return_argmin=True,
            rstate=np.random.default_rng(4),
        )

        time.sleep(2)
        self.check_cancellation_flags(spark_trials, FMIN_CANCELLED_REASON_TIMEOUT)

        # There are 2 finished trials, 1 cancelled running trial and 1 cancelled
        # new trial. We do not need to check the new trial since it is not started yet.
        self.assertGreaterEqual(
            spark_trials.count_successful_trials(),
            1,
            "Expected at least 1 successful trial but found none.",
        )
        self.assertGreaterEqual(
            spark_trials.count_cancelled_trials(),
            1,
            "Expected at least 1 cancelled trial but found none.",
        )

    def test_invalid_timeout(self):
        with self.assertRaisesRegexp(
            Exception,
            "timeout argument should be None or a positive value. Given value: -1",
        ):
            SparkTrials(parallelism=4, timeout=-1)
        with self.assertRaisesRegexp(
            Exception,
            "timeout argument should be None or a positive value. Given value: True",
        ):
            SparkTrials(parallelism=4, timeout=True)

    def test_exception_when_spark_not_available(self):
        import hyperopt

        orig_have_spark = hyperopt.spark._have_spark
        hyperopt.spark._have_spark = False
        try:
            with self.assertRaisesRegexp(Exception, "cannot import pyspark"):
                SparkTrials(parallelism=4)
        finally:
            hyperopt.spark._have_spark = orig_have_spark

    def test_broadcast_variable(self):
        # User code with pyspark broadcast variable.
        x = [1, 2, 3]
        bc_x = self.sc.broadcast(x)

        def fn_with_broadcast(params):
            local_x = bc_x.value[2]
            return {"loss": local_x, "status": "ok"}

        spark_trials = SparkTrials(parallelism=1)
        with patch_logger("hyperopt-spark", logging.DEBUG) as output:
            fmin(
                fn=fn_with_broadcast,
                space=hp.uniform("x", -0.25, 5),
                algo=anneal.suggest,
                max_evals=1,
                trials=spark_trials,
                max_queue_len=1,
                show_progressbar=False,
            )

            log_output = output.getvalue().strip()
            success_msg = "trial task 0 succeeded"
            self.assertIn(
                success_msg,
                log_output,
                "Expect fmin() with broadcast variable succeed."
                "Expected in logged output: {expected}\n"
                "Actual logged output: {actual}\n".format(
                    expected=success_msg, actual=log_output
                ),
            )

    def test_no_retry_for_long_tasks(self):
        NUM_TRIALS = 2
        output_dir = tempfile.mkdtemp()

        def fn(_):
            with open(os.path.join(output_dir, str(timeit.default_timer())), "w") as f:
                f.write("1")
            raise Exception("Failed!")

        spark_trials = SparkTrials(parallelism=2)
        try:
            fmin(
                fn=fn,
                space=hp.uniform("x", 0, 1),
                algo=anneal.suggest,
                max_evals=NUM_TRIALS,
                trials=spark_trials,
                show_progressbar=False,
                return_argmin=False,
            )
        except BaseException as e:
            self.assertEqual(
                "There are no evaluation tasks, cannot return argmin of task losses.",
                str(e),
            )

        call_count = len(os.listdir(output_dir))
        self.assertEqual(NUM_TRIALS, call_count)

    def test_cancel_button_with_job_cancellation(self):
        if not self.sparkSupportsJobCancelling():
            print(
                "Skipping cancel button test since this Apache PySpark version does not "
                "support cancelling jobs by job group ID."
            )
            return

        mp_manager = multiprocessing.Manager()
        ready_for_sigkill_event = mp_manager.Event()

        # Testing in a subprocess since sending SIGINT to the main process that runs all
        # tests might accidentally kill them all depending on timing.
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            pid = pool.submit(os.getpid).result()
            f = pool.submit(
                f_test_cancel_button_with_job_cancellation, ready_for_sigkill_event
            )
            # time.sleep(5)
            ready_for_sigkill_event.wait()
            os.kill(pid, signal.SIGINT)
            assert f.result(timeout=5) == True

    def test_early_stop(self):
        # stop after at least 3 trials succeed
        def early_stop_fn(trials):
            num_ok_trials = 0
            for trial in trials:
                if trial["result"]["status"] == STATUS_OK:
                    num_ok_trials += 1
            return num_ok_trials >= 3, []

        spark_trials = SparkTrials(parallelism=2)
        with patch_logger("hyperopt-spark", logging.DEBUG) as output:
            fmin(
                fn=lambda x: {"loss": x, "status": STATUS_OK, "other_info": "hello"},
                space=hp.uniform("x", -1, 1),
                algo=rand.suggest,
                max_evals=10,
                trials=spark_trials,
                early_stop_fn=early_stop_fn,
            )
            log_output = output.getvalue().strip()

            num_successful_trials = spark_trials.count_successful_trials()
            self.assertGreaterEqual(num_successful_trials, 3)
            self.assertLess(num_successful_trials, 10)

            self.assertIn(
                "fmin thread exits normally",
                log_output,
                f'Debug info "fmin thread exits normally" missing from log: {log_output}',
            )

            self.check_cancellation_flags(
                spark_trials, FMIN_CANCELLED_REASON_EARLY_STOPPING
            )
            self.assertIn(
                "fmin cancelled because of early stopping condition",
                log_output,
                f"Debug info for early stopping missing from log: {log_output}",
            )

    def test_pin_thread_off(self):
        if self._pin_mode_enabled:
            raise unittest.SkipTest()

        spark_trials = SparkTrials(parallelism=2)
        self.assertFalse(spark_trials._spark_pinned_threads_enabled)
        self.assertTrue(spark_trials._spark_supports_job_cancelling)
        fmin(
            fn=lambda x: x + 1,
            space=hp.uniform("x", -1, 1),
            algo=rand.suggest,
            max_evals=5,
            trials=spark_trials,
        )
        self.assertEqual(spark_trials.count_successful_trials(), 5)

    def test_pin_thread_on(self):
        if not self._pin_mode_enabled:
            raise unittest.SkipTest()

        spark_trials = SparkTrials(parallelism=2)
        self.assertTrue(spark_trials._spark_pinned_threads_enabled)
        self.assertTrue(spark_trials._spark_supports_job_cancelling)
        fmin(
            fn=lambda x: x + 1,
            space=hp.uniform("x", -1, 1),
            algo=rand.suggest,
            max_evals=5,
            trials=spark_trials,
        )
        self.assertEqual(spark_trials.count_successful_trials(), 5)


def f_test_cancel_button_with_job_cancellation(ready_for_sigkill_event):
    """
    This function is related to test case test_cancel_button_with_job_cancellation.

    This function needs to be at the module level, because otherwise it cannot be pickled
    and started as a separate process.

    :param ready_for_sigkill_event: Event that is triggered when at least one trial ran
    :return: True if successful
    """

    def fn(x):
        """Dummy function to optimize"""
        time.sleep(0.1)
        return x

    def early_stop(trials, *args, **kwargs):
        """
        Using the early_stop callback to trigger the sigkill event after we ran one trial successfully
        """
        if spark_trials.count_successful_trials() >= 1:
            ready_for_sigkill_event.set()
        return False, []

    spark_trials = SparkTrials(parallelism=4)
    try:
        fmin(
            fn=fn,
            space=hp.uniform("x", -1, 1),
            algo=rand.suggest,
            max_evals=100,
            trials=spark_trials,
            max_queue_len=2,
            show_progressbar=False,
            return_argmin=False,
            early_stop_fn=early_stop,
        )
        assert False, "KeyboardInterrupt was never thrown"
    except KeyboardInterrupt:
        pass

    # Verify that state is correctly handled
    assert (
        spark_trials.count_successful_trials() >= 1
    ), "There should be at least one successful trial"
    assert spark_trials.best_trial is not None, "There should be a best trial set"
    assert spark_trials._fmin_cancelled, "Cancelled should be set"
    assert (
        spark_trials._fmin_cancelled_reason == "fmin run cancelled by user"
    ), "Cancellation reason not set"

    return True
