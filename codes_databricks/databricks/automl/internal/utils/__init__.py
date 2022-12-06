from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from databricks.automl.internal.alerts import ExecutionTimeoutAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.errors import ExecutionTimeoutError


def time_check(start_time: datetime, timeout: Optional[int],
               alert_manager: AlertManager) -> Tuple[datetime, Optional[int]]:
    """
    Checks if the user has passed a non-null timeout and recalculates the new timeout
    based on the elapsed time. If the new timeout is <= 0, throw an ExecutionTimeoutError
    to indicate that we have reached a timeout

    :param start_time: Time when AutoML run started
    :param timeout: Optional timeout passed by the user
    :returns (Current time, Newly calculated timeout or None)
    """
    now = datetime.now()
    if not timeout:
        return now, timeout

    elapsed_time = int((now - start_time).total_seconds())
    new_timeout = timeout - elapsed_time
    if new_timeout <= 0:
        alert_manager.record(ExecutionTimeoutAlert())
        raise ExecutionTimeoutError(
            "Execution timed out before any trials could be successfully run. "
            "Please increase the timeout for AutoML to run some trials.")
    return now, new_timeout