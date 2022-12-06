from typing import List, Optional

from databricks.feature_store.entities.cloud import Cloud
from databricks.feature_store.entities.store_type import StoreType
from databricks.feature_store.online_store_spec import (
    OnlineStoreSpec,
)
from databricks.feature_store.online_store_spec.online_store_properties import (
    ACCOUNT_URI,
    CONTAINER_NAME,
    AZURE_COSMOSDB,
    AUTHORIZATION_KEY,
    DATABASE_NAME,
    SECRETS,
)
from databricks.feature_store.utils.uc_utils import LOCAL_METASTORE_NAMES


class AzureCosmosDBSpec(OnlineStoreSpec):
    """
    This :class:`OnlineStoreSpec <databricks.feature_store.online_store_spec.online_store_spec.OnlineStoreSpec>`
    implementation is intended for publishing features to Azure Cosmos DB.

    If `database_name` and `container_name` are not provided,
    :meth:`FeatureStoreClient.publish_table <databricks.feature_store.client.FeatureStoreClient.publish_table>`
    will use the offline store's database and table name as the Cosmos DB database and container name.

    The expected read or write secret for Cosmos DB for a given ``{prefix}`` string is
    ``${prefix}-authorization-key``.

    The authorization key can be either the Cosmos DB account primary or secondary key.

    .. note::
        Available in version >= 0.5.0.

    :param account_uri: URI of the Cosmos DB account.
    :param database_name: Database name.
    :param container_name: Container name.
    :param read_secret_prefix: Prefix for read secret.
    :param write_secret_prefix: Prefix for write secret.
    """

    def __init__(
        self,
        *,
        account_uri: str,
        database_name: Optional[str] = None,
        container_name: Optional[str] = None,
        read_secret_prefix: Optional[str] = None,
        write_secret_prefix: str,
    ):
        """Initialize AzureCosmosDBSpec object."""
        super().__init__(
            AZURE_COSMOSDB,
            database_name=database_name,
            read_secret_prefix=read_secret_prefix,
            write_secret_prefix=write_secret_prefix,
            _internal_properties={
                ACCOUNT_URI: account_uri,
                CONTAINER_NAME: container_name,
            },
        )

    @property
    def account_uri(self):
        """
        Account URI of the online store.
        """
        return self._properties[ACCOUNT_URI]

    @property
    def database_name(self):
        """
        Database name.
        """
        return self._properties[DATABASE_NAME]

    @property
    def container_name(self):
        """
        Container name.
        """
        return self._properties[CONTAINER_NAME]

    @property
    def cloud(self):
        """Define the cloud property for the data store."""
        return Cloud.AZURE

    @property
    def store_type(self):
        """Define the data store type."""
        return StoreType.COSMOSDB

    def _lookup_authorization_key_with_write_permissions(self) -> str:
        """
        Authorization key that has write access to the online store, resolved through the write_secret_prefix and dbutils.

        WARNING: do not hold onto the returned secret for longer than necessary, for example saving in
        data structures, files, other persistent backends. Use it only for directly accessing resources
        and then allow the Python VM to remove the reference as soon as it's out of scope.
        """
        return self._lookup_secret_with_write_permissions(AUTHORIZATION_KEY)

    def _validate_credentials(self):
        """
        Validate that the expected credentials were provided and are unambiguous.
        """
        # No validation is currently required since write_secret_prefix is a required parameter.
        pass

    def _valid_secret_suffixes(self) -> List[str]:
        """
        List of valid secret suffixes.
        """
        return [AUTHORIZATION_KEY]

    def _expected_secret_suffixes(self) -> List[str]:
        """
        List of expected secret suffixes.
        """
        return [AUTHORIZATION_KEY]

    def auth_type(self):
        """Publish Auth type."""
        return SECRETS

    def _augment_online_store_spec(self, full_feature_table_name):
        """
        Apply default database and table name for Azure Cosmos DB.
        Local workspace hive metastore: database = <database>, table = <table>
        UC: database = <catalog>.<database>, table = <table>
        """
        if (self.database_name is None) != (self.container_name is None):
            raise ValueError(
                f"The OnlineStoreSpec {self.store_type} must specify either both database_name "
                f"and container_name, or neither."
            )
        elif (self.database_name is None) and (self.container_name is None):
            catalog_name, database_name, table_name = full_feature_table_name.split(".")
            online_database_name = (
                f"{database_name}"
                if catalog_name in LOCAL_METASTORE_NAMES
                else f"{catalog_name}.{database_name}"
            )
            return self.clone(
                **{DATABASE_NAME: online_database_name, CONTAINER_NAME: table_name}
            )
        return self

    def _get_online_store_name(self):
        """
        Online store name for Azure Cosmos DB.
        """
        return f"{self.database_name}.{self.container_name}"
