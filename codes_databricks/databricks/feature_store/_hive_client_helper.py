import logging
from databricks.feature_store.hive_client import HiveClient
from databricks.feature_store.utils.uc_utils import LOCAL_METASTORE_NAMES

_logger = logging.getLogger(__name__)


class HiveClientHelper:
    """
    Helper functions that wrap calls to the hive client with application specific business logic.
    """

    def __init__(self, hive_client: HiveClient):
        self._hive_client = hive_client

    def _check_catalog_exists(self, full_table_name):
        """
        Check for the existence of a catalog.

        feature_table_name should have the form <catalog_name>.<database_name>.<table_name>.
        Check whether catalog_name is a catalog in the metastore.
        """
        catalog_name, database_name, table_name = full_table_name.split(".")
        # Local metastore is not an actual catalog.
        if catalog_name in LOCAL_METASTORE_NAMES:
            return True
        if not self._hive_client.catalog_exists(catalog_name):
            raise ValueError(
                f"Catalog '{catalog_name}' does not exist in the metastore."
            )

    def _check_database_exists(self, full_table_name):
        """
        Check for the existence of a database.

        feature_table_name should have the form <catalog_name>.<database_name>.<table_name>.
        Check whether database_name is a database in the catalog.
        """
        catalog_name, database_name, table_name = full_table_name.split(".")
        if not self._hive_client.database_exists(catalog_name, database_name):
            raise ValueError(
                f"Database '{database_name}' does not exist in catalog '{catalog_name}'."
            )

    def check_catalog_database_exists(self, full_table_name):
        self._check_catalog_exists(full_table_name)
        self._check_database_exists(full_table_name)

    def check_feature_table_exists(self, full_table_name):
        self.check_catalog_database_exists(full_table_name)
        if not self._hive_client.table_exists(full_table_name):
            raise ValueError(
                f"The feature data table for '{full_table_name}' could not be found."
            )
