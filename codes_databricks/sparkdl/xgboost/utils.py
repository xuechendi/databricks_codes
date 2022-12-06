#
# DATABRICKS CONFIDENTIAL & PROPRIETARY
# __________________
#
# Copyright 2020 Databricks, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property of Databricks, Inc.
# and its suppliers, if any.  The intellectual and technical concepts contained herein are
# proprietary to Databricks, Inc. and its suppliers and may be covered by U.S. and foreign Patents,
# patents in process, and are protected by trade secret and/or copyright law. Dissemination, use,
# or reproduction of this information is strictly forbidden unless prior written permission is
# obtained from Databricks, Inc.
#
# If you view or obtain a copy of this information and believe Databricks, Inc. may not have
# intended it to be made available, please promptly report it to Databricks Legal Department
# @ legal@databricks.com.
#

import inspect
from threading import Thread
from xgboost import rabit
from xgboost.tracker import RabitTracker
import pyspark
from pyspark.sql.session import SparkSession
from pyspark.ml.param.shared import Param, Params


def get_class_name(cls):
    return f"{cls.__module__}.{cls.__name__}"


def _get_default_params_from_func(func, unsupported_set):
    """
    Returns a dictionary of parameters and their default value of function fn.
    Only the parameters with a default value will be included.
    """
    sig = inspect.signature(func)
    filtered_params_dict = dict()
    for parameter in sig.parameters.values():
        # Remove parameters without a default value and those in the unsupported_set
        if parameter.default is not parameter.empty \
                and parameter.name not in unsupported_set:
            filtered_params_dict[parameter.name] = parameter.default
    return filtered_params_dict


class HasArbitraryParamsDict(Params):
    """
    This is a Params based class that is extended by _XGBoostParams
    and holds the variable to store the **kwargs parts of the XGBoost
    input.
    """

    arbitraryParamsDict = Param(Params._dummy(), "arbitraryParamsDict", \
        "This parameter holds all of the user defined parameters that the sklearn" + \
        "implementation of XGBoost can't recognize. It is stored as a dictionary.")

    def setArbitraryParamsDict(self, value):
        return self._set(arbitraryParamsDict=value)

    def getArbitraryParamsDict(self, value):
        return self.getOrDefault(self.arbitraryParamsDict)


class HasBaseMarginCol(Params):
    """
    This is a Params based class that is extended by _XGBoostParams
    and holds the variable to store the base margin column part of XGboost.
    """
    baseMarginCol = Param(
        Params._dummy(), "baseMarginCol",
        "This stores the name for the column of the base margin")

    def setBaseMarginCol(self, value):
        return self._set(baseMarginCol=value)

    def getBaseMarginCol(self, value):
        return self.getOrDefault(self.baseMarginCol)


class RabitContext:
    """
    A context controlling rabit initialization and finalization.
    This isn't specificially necessary (note Part 3), but it is more understandable coding-wise.
    """
    def __init__(self, args, context):
        self.args = args
        self.args.append(
            ('DMLC_TASK_ID=' + str(context.partitionId())).encode())

    def __enter__(self):
        rabit.init(self.args)

    def __exit__(self, *args):
        rabit.finalize()


def _start_tracker(context, n_workers):
    """
    Start Rabit tracker with n_workers
    """
    env = {'DMLC_NUM_WORKER': n_workers}
    host = get_host_ip(context)
    rabit_context = RabitTracker(host_ip=host, n_workers=n_workers)
    env.update(rabit_context.worker_envs())
    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join)
    thread.daemon = True
    thread.start()
    return env


def _get_rabit_args(context, n_workers):
    """
    Get rabit context arguments to send to each worker.
    """
    env = _start_tracker(context, n_workers)
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]
    return rabit_args


def get_host_ip(context):
    """
    Gets the hostIP for Spark. This essentially gets the IP of the first worker.
    """
    task_ip_list = [
        info.address.split(":")[0] for info in context.getTaskInfos()
    ]
    return task_ip_list[0]


def _get_args_from_message_list(messages):
    """
    A function to send/recieve messages in barrier context mode
    """
    output = ""
    for message in messages:
        if message != "":
            output = message
            break
    return [
        elem.split("'")[1].encode() for elem in output.strip('][').split(', ')
    ]


def _get_spark_session():
    """Get or create spark session. Note: This function can only be invoked from driver side."""
    if pyspark.TaskContext.get() is not None:
        # This is a safety check.
        raise RuntimeError(
            '_get_spark_session should not be invoked from executor side.')
    return SparkSession.builder.getOrCreate()
