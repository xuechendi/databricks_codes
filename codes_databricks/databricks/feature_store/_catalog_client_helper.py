import logging
from typing import Set

from databricks.feature_store.catalog_client import CatalogClient
from databricks.feature_store.databricks_client import DatabricksClient
from databricks.feature_store.utils import utils
from databricks.feature_store.utils.request_context import (
    RequestContext,
)
from mlflow.utils import databricks_utils

_logger = logging.getLogger(__name__)


class CatalogClientHelper:
    """
    Helper functions that wrap calls to the catalog client with additional business logic, possibly invoking
    other clients as well (eg, DatabricksClient).
    """

    def __init__(
        self, catalog_client: CatalogClient, databricks_client: DatabricksClient
    ):
        self._catalog_client = catalog_client
        self._databricks_client = databricks_client

    def add_producers(
        self, feature_table_name, producer_action, req_context: RequestContext
    ):
        try:
            if utils.is_in_databricks_job():
                job_id = databricks_utils.get_job_id()
                if job_id:
                    job_id = int(job_id)
                    job_run_id = databricks_utils.get_job_run_id()
                    job_run_id = int(job_run_id) if job_run_id else None
                    self._catalog_client.add_job_producer(
                        feature_table_name,
                        job_id,
                        job_run_id,
                        producer_action,
                        req_context,
                    )
                else:
                    _logger.warning(
                        f"Failed to record producer in the catalog. Missing job_id ({job_id})"
                    )
            elif databricks_utils.is_in_databricks_notebook():
                notebook_path = databricks_utils.get_notebook_path()
                notebook_id = databricks_utils.get_notebook_id()
                if notebook_id:
                    notebook_id = int(notebook_id)
                    revision_id = self._databricks_client.take_notebook_snapshot(
                        notebook_path
                    )
                    revision_id = int(revision_id) if revision_id else None
                    self._catalog_client.add_notebook_producer(
                        feature_table_name,
                        notebook_id,
                        revision_id,
                        producer_action,
                        req_context,
                    )
                else:
                    _logger.warning(
                        f"Failed to record producer in the catalog. "
                        f"Missing notebook_id ({notebook_id})."
                    )
        except Exception as e:
            _logger.warning(
                f"Failed to record producer in the catalog. Exception: {e}",
                exc_info=True,
            )

    def add_consumers(self, feature_table_map, req_context: RequestContext):
        try:
            if utils.is_in_databricks_job():
                job_id = databricks_utils.get_job_id()
                job_run_id = databricks_utils.get_job_run_id()
                if job_id:
                    job_id = int(job_id)
                    job_run_id = int(job_run_id) if job_run_id else None
                    self._catalog_client.add_job_consumer(
                        feature_table_map, job_id, job_run_id, req_context
                    )
                else:
                    _logger.warning(
                        f"Failed to record consumer in the catalog. Missing job_run_id ({job_id})."
                    )
            elif databricks_utils.is_in_databricks_notebook():
                notebook_path = databricks_utils.get_notebook_path()
                notebook_id = databricks_utils.get_notebook_id()
                if notebook_id:
                    notebook_id = int(notebook_id)
                    revision_id = self._databricks_client.take_notebook_snapshot(
                        notebook_path
                    )
                    revision_id = int(revision_id) if revision_id else None
                    self._catalog_client.add_notebook_consumer(
                        feature_table_map, notebook_id, revision_id, req_context
                    )
                else:
                    _logger.warning(
                        f"Failed to record consumer in the catalog. "
                        f"Missing notebook_id ({notebook_id})."
                    )
        except Exception as e:
            _logger.warning(
                f"Failed to record consumer in the catalog. Exception: {e}",
                exc_info=True,
            )

    def add_data_sources(
        self,
        name: str,
        tables: Set[str],
        paths: Set[str],
        custom_sources: Set[str],
        req_context: RequestContext,
    ):
        try:
            self._catalog_client.add_data_sources(
                name,
                tables=list(tables),
                paths=list(paths),
                custom_sources=list(custom_sources),
                req_context=req_context,
            )
        except Exception as e:
            _logger.warning(
                f"Failed to record data sources in the catalog. Exception: {e}",
                exc_info=True,
            )
