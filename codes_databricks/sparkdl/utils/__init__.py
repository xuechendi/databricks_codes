#
# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
import logging


def _getConfBoolean(sqlContext, key, defaultValue):
    """
    Get the conf "key" from the given sqlContext,
    or return the default value if the conf is not set.
    This expects the conf value to be a boolean or string; if the value is a string,
    this checks for all capitalization patterns of "true" and "false" to match Scala.
    :param key: string for conf name
    """
    # Convert default value to str to avoid a Spark 2.3.1 + Python 3 bug: SPARK-25397
    val = sqlContext.getConf(key, str(defaultValue))
    # Convert val to str to handle unicode issues across Python 2 and 3.
    lowercase_val = str(val.lower())
    if lowercase_val == 'true':
        return True
    elif lowercase_val == 'false':
        return False
    else:
        raise Exception("_getConfBoolean expected a boolean conf value but found value of type {} "
                        "with value: {}".format(type(val), val))


def get_logger(name, level='INFO'):
    """ Gets a logger by name, or creates and configures it for the first time. """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # If the logger is configured, skip the configure
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    return logger


def _get_max_num_concurrent_tasks(sc):
    """Gets the current max number of concurrent tasks."""
    # spark 3.1 and above has a different API for fetching max concurrent tasks
    if sc._jsc.sc().version() >= '3.1':
        return sc._jsc.sc().maxNumConcurrentTasks(
            sc._jsc.sc().resourceProfileManager().resourceProfileFromId(0)
        )
    return sc._jsc.sc().maxNumConcurrentTasks()
