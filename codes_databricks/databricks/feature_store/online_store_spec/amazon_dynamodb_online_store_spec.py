from datetime import timedelta
from typing import Union, List, Optional

from mlflow.utils.annotations import deprecated
from pyspark.sql.utils import IllegalArgumentException

from databricks.feature_store.entities.cloud import Cloud
from databricks.feature_store.entities.store_type import StoreType
from databricks.feature_store.online_store_spec import (
    OnlineStoreSpec,
)
from databricks.feature_store.online_store_spec.online_store_properties import (
    AWS_DYNAMODB,
    REGION,
    SESSION_TOKEN,
    SECRET_ACCESS_KEY,
    ACCESS_KEY_ID,
    TTL,
    SECRETS,
    ROLE,
    TABLE_NAME,
)
from databricks.feature_store.utils.utils import is_empty
from databricks.feature_store.utils.uc_utils import reformat_full_table_name
import logging

_logger = logging.getLogger(__name__)


class AmazonDynamoDBSpec(OnlineStoreSpec):
    """
    This :class:`OnlineStoreSpec <databricks.feature_store.online_store_spec.online_store_spec.OnlineStoreSpec>`
    implementation is intended for publishing features to Amazon DynamoDB.

    If `table_name` is not provided,
    :meth:`FeatureStoreClient.publish_table <databricks.feature_store.client.FeatureStoreClient.publish_table>`
    will use the offline store's database and table name combined as the online table name.

    To use a different table name in the online store, provide a value for the `table_name` argument.

    The expected read or write secrets for DynamoDB for a given ``{prefix}`` string are
    ``${prefix}-access-key-id``, ``${prefix}-secret-access-key``, and ``${prefix}-session-token``.

    If none of the access_key_id, secret_access_key, and write_secret_prefix are passed,
    the instance profile attached to the cluster will be used to write to DynamoDB.

    .. note::

      AmazonDynamoDBSpec is available in version >= 0.3.8.

      Instance profile based writes are available in version >= 0.4.1.

    :param region: Region to access online store.
    :param access_key_id: Access key ID that has access to the online store. **Deprecated** as of version 0.6.0.
      Use ``write_secret_prefix`` instead.
    :param secret_access_key: Secret access key to access the online store. **Deprecated** as of version 0.6.0.
      Use ``write_secret_prefix`` instead.
    :param session_token: Session token to access the online store. **Deprecated** as of version 0.6.0.
      Use ``write_secret_prefix`` instead.
    :param table_name: Table name.
    :param read_secret_prefix: Prefix for read secret.
    :param write_secret_prefix: Prefix for write secret.
    :param ttl: The time to live for data published to the online store. This attribute is only applicable when
      publishing time series feature tables. If the time to live is specified for a time series table,
      :meth:`FeatureStoreClient.publish_table` will publish a window of data instead of the latest snapshot.
    """

    # TODO (ML-23105): Remove explicit parameters for MLR 12.0.
    def __init__(
        self,
        *,
        region: Union[str, None],
        access_key_id: Union[str, None] = None,
        secret_access_key: Union[str, None] = None,
        session_token: Union[str, None] = None,
        table_name: Union[str, None] = None,
        read_secret_prefix: Union[str, None] = None,
        write_secret_prefix: Union[str, None] = None,
        ttl: Optional[timedelta] = None,
    ):
        """Initialize AmazonDynamoDBSpec object."""
        super().__init__(
            AWS_DYNAMODB,
            table_name=table_name,
            read_secret_prefix=read_secret_prefix,
            write_secret_prefix=write_secret_prefix,
            _internal_properties={
                REGION: region,
                ACCESS_KEY_ID: access_key_id,
                SECRET_ACCESS_KEY: secret_access_key,
                SESSION_TOKEN: session_token,
                TTL: ttl,
            },
        )

    @property
    @deprecated(alternative="write_secret_prefix", since="v0.6.0")
    def access_key_id(self):
        """
        Access key ID that has access to the online store.
        Property will be empty if ``write_secret_prefix`` or the
        instance profile attached to the cluster are intended to be used.
        """
        return self._properties[ACCESS_KEY_ID]

    @property
    @deprecated(alternative="write_secret_prefix", since="v0.6.0")
    def secret_access_key(self):
        """
        Secret access key to access the online store.
        Property will be empty if ``write_secret_prefix`` or the
        instance profile attached to the cluster are intended to be used.
        """
        return self._properties[SECRET_ACCESS_KEY]

    @property
    @deprecated(alternative="write_secret_prefix", since="v0.6.0")
    def session_token(self):
        """
        Session token to access the online store.
        Property will be empty if ``write_secret_prefix`` or the
        instance profile attached to the cluster are intended to be used.
        """
        return self._properties[SESSION_TOKEN]

    @property
    def cloud(self):
        """Define the cloud property for the data store."""
        return Cloud.AWS

    @property
    def store_type(self):
        """Define the data store type."""
        return StoreType.DYNAMODB

    @property
    def region(self):
        """Region to access the online store."""
        return self._properties[REGION]

    @property
    def ttl(self) -> Optional[timedelta]:
        """Time to live attribute for the online store."""
        return self._properties[TTL]

    def auth_type(self):
        """Publish Auth type."""
        return ROLE if self._is_role_based() else SECRETS

    def _is_role_based(self) -> bool:
        return (
            is_empty(self.access_key_id)
            and is_empty(self.secret_access_key)
            and is_empty(self.session_token)
            and is_empty(self.write_secret_prefix)
        )

    def _lookup_access_key_id_with_write_permissions(self) -> str:
        """
        Access key ID that has write access to the online store, resolved through the write_secret_prefix and dbutils.

        WARNING: do not hold onto the returned secret for longer than necessary, for example saving in
        data structures, files, other persistent backends. Use it only for directly accessing resources
        and then allow the Python VM to remove the reference as soon as it's out of scope.
        """
        return self._lookup_secret_with_write_permissions(ACCESS_KEY_ID)

    def _lookup_secret_access_key_with_write_permissions(self) -> str:
        """
        Secret access key that has write access to the online store, resolved through the write_secret_prefix and dbutils.

        WARNING: do not hold onto the returned secret for longer than necessary, for example saving in
        data structures, files, other persistent backends. Use it only for directly accessing resources
        and then allow the Python VM to remove the reference as soon as it's out of scope.
        """
        return self._lookup_secret_with_write_permissions(SECRET_ACCESS_KEY)

    def _lookup_session_token_with_write_permissions(self) -> Union[str, None]:
        """
        Optional session token that has write access to the online store, resolved through the write_secret_prefix and dbutils.

        WARNING: do not hold onto the returned secret for longer than necessary, for example saving in
        data structures, files, other persistent backends. Use it only for directly accessing resources
        and then allow the Python VM to remove the reference as soon as it's out of scope.
        """
        # An invalid scope/key throws: "IllegalArgumentException: Secret does not exist with scope: X and key: Y"
        try:
            return self._lookup_secret_with_write_permissions(SESSION_TOKEN)
        except IllegalArgumentException:
            return None

    def _validate_credentials(self):
        """
        Validate that the expected credentials were provided and are unambiguous.
        """
        # Validate that the user passed either both (access_key_id, secret_access_key) or a write_secret_prefix,
        # else assume the user intends to use attached instance profile in the cluster.
        # TODO (ML-19653): Align the allowed credentials and error messages

        # If all write secret fields are empty, assume using role based publish.
        if self._is_role_based():
            _logger.info(
                "No explicit credentials for write provided. "
                "Instance profile attached to the cluster will be used."
            )
            return
        # Validate that user didn't pass in any of (access_key_id, secret_access_key, session_token)
        # AND write_secret_prefix, else throw an error because this is ambiguous
        if self.write_secret_prefix is not None:
            if not is_empty(self.access_key_id):
                raise Exception(
                    "Use either 'access_key_id' or 'write_secret_prefix', but not both."
                )
            if not is_empty(self.secret_access_key):
                raise Exception(
                    "Use either 'secret_access_key' or 'write_secret_prefix', but not both."
                )
            if not is_empty(self.session_token):
                raise Exception(
                    "Use either 'session_token' or 'write_secret_prefix', but not both."
                )
        else:
            if is_empty(self.access_key_id) != is_empty(self.secret_access_key):
                raise Exception(
                    "Both 'access_key_id' and 'secret_access_key' needs to be provided."
                )
            if (
                is_empty(self.access_key_id)
                and is_empty(self.secret_access_key)
                and not is_empty(self.session_token)
            ):
                raise Exception(
                    "'session_token' can only be used along with both 'access_key_id' and 'secret_access_key'."
                )

    def _valid_secret_suffixes(self) -> List[str]:
        """
        List of valid secret suffixes.
        """
        return [ACCESS_KEY_ID, SECRET_ACCESS_KEY, SESSION_TOKEN]

    def _expected_secret_suffixes(self) -> List[str]:
        """
        List of expected secret suffixes.
        """
        return [ACCESS_KEY_ID, SECRET_ACCESS_KEY]

    def _augment_online_store_spec(self, full_feature_table_name):
        """
        Apply default table name for Amazon DynamoDB.
        Local workspace hive metastore: <database>.<table>
        UC: <catalog>.<database>.<table>
        """
        if self.table_name is None:
            return self.clone(
                **{TABLE_NAME: reformat_full_table_name(full_feature_table_name)}
            )
        return self

    def _get_online_store_name(self):
        """
        Online store name for Amazon DynamoDB.
        """
        return self.table_name
