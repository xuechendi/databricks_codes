# Copyright 2018 Databricks, Inc.

import logging
import traceback

import wrapt

from pyspark import SparkContext


@wrapt.decorator
def instrumented(func, self, args, kwargs):
    """
        A decorator that log the start and end of the function run
    """
    if SparkContext._active_spark_context is None:
        logging.info("No SparkContext exists. Not logging the training start and end time.")
        return func(*args, **kwargs)
    else:
        sc = SparkContext._active_spark_context
    instr = sc._jvm.org.apache.spark.ml.util.Instrumentation()
    instr.logNamedValue('mlModelClass',
                        "{0}.{1}".format(self.__class__.__module__,
                                         self.__class__.__name__))
    try:
        return_val = func(*args, **kwargs)
    except Exception:
        error_string = traceback.format_exc()
        # TODO(SPARK-27657): logFailure swallows error_string
        instr.logFailure(sc._jvm.Exception(error_string))
        raise
    else:
        instr.logSuccess()
        return return_val
