from __future__ import annotations
import json
import logging

from databricks.automl.shared.errors import AutomlError
from mlflow.utils import databricks_utils as mlflow_databricks_utils

_logger = logging.getLogger(__name__)


class DatabricksUtils:
    # Used to identify if this is run by a genie user who doesn't have a home directory
    # and hence fall back to using "/" as the automl root directory
    DBADMIN_USER_SUFFIX = "+dbadmin@databricks.com"

    def __init__(self,
                 user: str,
                 api_url: str,
                 api_token: str,
                 cluster_id: str,
                 browser_host_name: Optional[str] = None,
                 org_id: Optional[int] = None,
                 driver_notebook_path: Optional[str] = None,
                 job_run_id: Optional[int] = None):
        """
        This is a wrapper class for the information fetched from the databricks spark context
        and contains only the fields and methods that are required by the databricks context.
        This is necessary for allowing databricks context to generate, execute and write notebooks
        to the databricks workspace without having to rely on the original databricks spark context
        since the spark context on workers is manually created and not the same one present on the driver

        :param user: The name of the user
        :param api_url: Url to make API calls
        :param api_token: Token to pass when making API calls
        :param host_name: Hostname for browser
        :param driver_notebook_path: Path of the driver notebook where AutoML was called
        :param job_run_id: Databricks Job Run ID, only set if executing in a job
        """
        self._user = user
        self._api_url = api_url
        self._api_token = api_token
        self._cluster_id = cluster_id
        self._job_run_id = job_run_id
        if browser_host_name and org_id:
            self._host_name = f"https://{browser_host_name}/?o={org_id}"
        elif browser_host_name:
            self._host_name = f"https://{browser_host_name}"
        else:
            self._host_name = None

        self._driver_notebook_path = driver_notebook_path

    @staticmethod
    def create() -> DatabrickUtils:
        try:
            dbutils = mlflow_databricks_utils._get_dbutils()
            ctx = dbutils.entry_point.getDbutils().notebook().getContext()

            user = ctx.userName().get()
            api_url = ctx.apiUrl().get()
            api_token = ctx.apiToken().get()
            cluster_id = ctx.clusterId().get()
            browser_host_name = ctx.browserHostName().get() if ctx.browserHostName().isDefined(
            ) else None
            org_id = ctx.workspaceId().get() if ctx.workspaceId().isDefined() else None
            driver_notebook_path = ctx.notebookPath().get() if ctx.notebookPath().isDefined(
            ) else None
            job_run_id = ctx.idInJob().get() if ctx.idInJob().isDefined() else None
        except Exception as e:
            raise AutomlError("Unable to access databricks utils") from e

        return DatabricksUtils(
            user=user,
            api_url=api_url,
            api_token=api_token,
            cluster_id=cluster_id,
            browser_host_name=browser_host_name,
            org_id=org_id,
            driver_notebook_path=driver_notebook_path,
            job_run_id=job_run_id)

    @property
    def user(self) -> str:
        return self._user

    @property
    def is_user_dbadmin(self) -> bool:
        return self._user.endswith(self.DBADMIN_USER_SUFFIX)

    @property
    def api_url(self) -> str:
        return self._api_url

    @property
    def api_token(self) -> str:
        return self._api_token

    @property
    def cluster_id(self) -> str:
        return self._cluster_id

    @property
    def host_name(self) -> Optional[str]:
        return self._host_name

    @property
    def job_run_id(self) -> Optional[str]:
        return self._job_run_id

    @property
    def driver_notebook_path(self) -> Optional[str]:
        return self._driver_notebook_path

    def get_experiment_url(self, experiment_id: str, absolute: bool) -> str:
        url = f"#mlflow/experiments/{experiment_id}"
        if absolute:
            return self.to_absolute_url(url)
        return url

    def to_absolute_url(self, relative_url: str) -> str:
        if self._host_name:
            return f"{self._host_name}{relative_url}"
        else:
            _logger.warn("No host name to create absolute URL")
            return relative_url

    @staticmethod
    def display_html(html) -> None:
        mlflow_databricks_utils._get_dbutils().notebook.displayHTML(html)

    @staticmethod
    def run_notebook(path: str, timeout: Optional[int]) -> str:
        return mlflow_databricks_utils._get_dbutils().notebook.run(
            path, timeout_seconds=timeout or 0)
