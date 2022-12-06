import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import mlflow
from mlflow.exceptions import RestException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.utils import databricks_utils

from databricks.feature_store.constants import MODEL_DATA_PATH_ROOT
from databricks.feature_store.entities.cloud import Cloud
from databricks.feature_store.entities.store_type import StoreType
from databricks.feature_store.utils.utils_common import is_artifact_uri


def as_list(obj, default=None):
    if not obj:
        return default
    elif isinstance(obj, list):
        return obj
    else:
        return [obj]


def as_directory(path):
    if path.endswith("/"):
        return path
    return f"{path}/"


def is_empty(target: str):
    return target is None or len(target.strip()) == 0


class _NoDbutilsError(Exception):
    pass


def _get_dbutils():
    try:
        import IPython

        ip_shell = IPython.get_ipython()
        if ip_shell is None:
            raise _NoDbutilsError
        return ip_shell.ns_table["user_global"]["dbutils"]
    except ImportError:
        raise _NoDbutilsError
    except KeyError:
        raise _NoDbutilsError


def get_canonical_online_store_name(online_store):
    if online_store.cloud == Cloud.AWS:
        if online_store.store_type == StoreType.MYSQL:
            return "Amazon RDS MySQL"
        elif online_store.store_type == StoreType.AURORA_MYSQL:
            return "Amazon Aurora (MySQL-compatible)"
        elif online_store.store_type == StoreType.DYNAMODB:
            return "Amazon DynamoDB"
    elif online_store.cloud == Cloud.AZURE:
        if online_store.store_type == StoreType.MYSQL:
            return "Azure Database for MySQL"
        elif online_store.store_type == StoreType.SQL_SERVER:
            return "Azure SQL Database"


def utc_timestamp_ms_from_iso_datetime_string(date_string):
    # Python uses seconds for its time granularity, so we multiply by 1000 to convert to milliseconds.
    # The Feature Store backend returns timestamps in milliseconds, so this allows for direct comparisons.
    dt = datetime.fromisoformat(date_string)
    utc_dt = dt.replace(tzinfo=timezone.utc)
    return 1000 * utc_dt.timestamp()


def pip_depependency_pinned_major_version(pip_package_name, major_version):
    """
    Generate a pip dependency string that is pinned to a major version, for example: "databricks-feature-lookup==0.*"
    """
    return f"{pip_package_name}=={major_version}.*"


def add_mlflow_pip_depependency(conda_env, pip_package_name):
    """
    Add a new pip dependency to the conda environment taken from the raw MLflow model.  This method should only be
    called for conda environments created by MLflow rather than for generic conda environments, because it assumes
    the conda environment already contains pip as a dependency.  In the case of MLflow models, this is a safe
    assumption because MLflow always needs to add "mlflow" to the conda environment's pip dependencies.

    This is idempotent and will not add a pip package that is already present in the list of pip packages.
    """
    if pip_package_name is None or len(pip_package_name) == 0:
        raise ValueError(
            "Unexpected input: missing or empty pip_package_name parameter"
        )

    found_pip_dependency = False
    if conda_env is not None:
        for dep in conda_env["dependencies"]:
            if isinstance(dep, dict) and "pip" in dep:
                found_pip_dependency = True
                pip_deps = dep["pip"]
                if pip_package_name not in pip_deps:
                    pip_deps.append(pip_package_name)
        # Fail early rather than at model inference time
        if "dependencies" in conda_env and not found_pip_dependency:
            raise ValueError(
                "Unexpected input: mlflow conda_env did not contain pip as a dependency"
            )


def download_model_artifacts(model_uri, dir):
    """
    Downloads model artifacts from model_uri to dir. Intended for use only with Feature Store packaged models.

    :param model_uri: The location, in URI format, of a model. Must be either in the model registry
      (``models:/<model_name>/<model_version>``, ``models:/<model_name>/<stage>``) or the MLflow
      artifact store (``runs:/<mlflow_run_id>/run-relative/path/to/model``).
    :param dir: Location to place downloaded model artifacts.
    """
    if not is_artifact_uri(model_uri):
        raise ValueError(
            f"Invalid model URI '{model_uri}'."
            f"Use ``models:/model_name>/<version_number>`` or "
            f"``runs:/<mlflow_run_id>/run-relative/path/to/model``."
        )

    try:
        repo = get_artifact_repository(model_uri)
    except RestException as e:
        raise ValueError(f"The model at '{model_uri}' does not exist.", e)

    artifact_path = os.path.join(mlflow.pyfunc.DATA, MODEL_DATA_PATH_ROOT)
    if len(repo.list_artifacts(artifact_path)) == 0:
        raise ValueError(
            f"No suitable model found at '{model_uri}'. Either no model exists in this "
            f"artifact location or an existing model was not packaged with Feature Store metadata. "
            f"Only models logged by FeatureStoreClient.log_model can be used in inference."
        )

    return repo.download_artifacts(artifact_path="", dst_path=dir)


def validate_params_non_empty(params: Dict[str, Any], expected_params: List[str]):
    """
    Validate that none of the expected parameters are empty, otherwise raise a Value error
    for the first encountered empty parameter.

    Tested with the following param types:

    - str
    - Dict
    - List

    :param params: A dictionary of param names -> param values, for example as returned by locals()
    :param expected_params: List of params to check as non_empty
    """
    for expected_param in expected_params:
        if expected_param not in params:
            raise ValueError(
                f'Internal error: expected parameter "{expected_param}" not found in params dictionary'
            )
        param_value = params[expected_param]
        if not param_value:
            raise ValueError(f'Parameter "{expected_param}" cannot be empty')


def is_in_databricks_job():
    """
    Overrides the behavior of the mlflow databricks_utils.is_in_databricks_job() to account for the fact that
    some jobs have job_id but no run_id, for example one-time job runs.
    """
    try:
        return databricks_utils.get_job_id() is not None
    except Exception:
        return False


def is_in_databricks_env():
    """
    Determine if we are running in a Databricks environment (DBR, MLR, DLT, DCS, Mlflow Projects, Run Cmd 1.2 API, etc)

    If any invoked methods raise an exception, swallow the exception and return False out of an abundance of caution.
    """
    try:
        return (
            is_in_databricks_job()
            or databricks_utils.is_in_databricks_notebook()
            or databricks_utils.is_in_databricks_runtime()
        )
    except Exception:
        return False
