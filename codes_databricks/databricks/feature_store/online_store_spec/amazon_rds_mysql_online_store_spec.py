from databricks.feature_store.online_store_spec import (
    OnlineStoreSpec,
)
from databricks.feature_store.online_store_spec.online_store_properties import (
    USER,
    AWS_MYSQL,
    AWS_AURORA,
    HOSTNAME,
    PORT,
    DATABASE_NAME,
    TABLE_NAME,
    SECRETS,
)
from databricks.feature_store.entities.store_type import StoreType
from databricks.feature_store.entities.cloud import Cloud
from databricks.feature_store.utils.uc_utils import LOCAL_METASTORE_NAMES
from typing import Union


class AmazonRdsMySqlSpec(OnlineStoreSpec):
    """
    Class that defines and creates :class:`AmazonRdsMySqlSpec` objects.

    This :class:`OnlineStoreSpec` implementation is intended for publishing
    features to Amazon RDS MySQL and Aurora (MySQL-compatible edition).

    See :class:`OnlineStoreSpec` documentation for more usage information,
    including parameter descriptions.

    :param hostname: Hostname to access online store.
    :param port: Port number to access online store.
    :param user: Username that has access to the online store. **Deprecated** as of version 0.6.0.
      Use ``write_secret_prefix`` instead.
    :param password: Password to access the online store. **Deprecated** as of version 0.6.0.
      Use ``write_secret_prefix`` instead.
    :param database_name: Database name.
    :param table_name: Table name.
    :param driver_name: Name of custom JDBC driver to access the online store.
    :param read_secret_prefix: Prefix for read secret.
    :param write_secret_prefix: Prefix for write secret.

    .. todo::

       [ML-15546]: Identify clear mechanism to inherit constructor
       pydocs from base class and
       remove ``See xxx documentation for more usage information`` section.
    """

    # TODO (ML-23105): Remove explicit parameters for MLR 12.0.
    def __init__(
        self,
        hostname: str,
        port: int,
        user: Union[str, None] = None,
        password: Union[str, None] = None,
        database_name: Union[str, None] = None,
        table_name: Union[str, None] = None,
        driver_name: Union[str, None] = None,
        read_secret_prefix: Union[str, None] = None,
        write_secret_prefix: Union[str, None] = None,
    ):
        """Initialize AmazonRdsMySqlSpec objects."""
        super().__init__(
            AWS_MYSQL,
            hostname,
            port,
            user,
            password,
            database_name=database_name,
            table_name=table_name,
            driver_name=driver_name,
            read_secret_prefix=read_secret_prefix,
            write_secret_prefix=write_secret_prefix,
        )

    @property
    def hostname(self):
        """Hostname to access the online store."""
        return self._properties[HOSTNAME]

    @property
    def port(self):
        """Port number to access the online store."""
        return self._properties[PORT]

    @property
    def database_name(self):
        """Database name."""
        return self._properties[DATABASE_NAME]

    @property
    def cloud(self):
        """Define the cloud propert for the data store."""
        return Cloud.AWS

    @property
    def store_type(self):
        """Define the data store type property.

        .. todo::

           (mparkhe): Get the right ``_type``
        """
        if self.type == AWS_MYSQL:
            return StoreType.MYSQL
        elif self.type == AWS_AURORA:
            return StoreType.AURORA_MYSQL

    def auth_type(self):
        """Publish Auth type."""
        return SECRETS

    def _augment_online_store_spec(self, full_feature_table_name):
        """
        Apply default database and table name for Amazon RDS MySQL.
        Local workspace hive metastore: database = <database>, table = <table>
        UC: database = <catalog>-<database>, table = <table>
        """
        if (self.database_name is None) != (self.table_name is None):
            raise ValueError(
                f"The OnlineStoreSpec {self.store_type} must specify either both database_name "
                f"and table_name, or neither."
            )
        elif (self.database_name is None) and (self.table_name is None):
            catalog_name, database_name, table_name = full_feature_table_name.split(".")
            online_database_name = (
                f"{database_name}"
                if catalog_name in LOCAL_METASTORE_NAMES
                else f"{catalog_name}-{database_name}"
            )
            return self.clone(
                **{DATABASE_NAME: online_database_name, TABLE_NAME: table_name}
            )
        return self

    def _get_online_store_name(self):
        """
        Online store name for Amazon RDS MySQL
        """
        return f"{self.database_name}.{self.table_name}"
