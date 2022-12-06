from databricks.feature_store.online_store_publish_client import (
    OnlineStorePublishClient,
    is_rdbms_spec,
    is_nosql_spec,
)
from databricks.feature_store.online_store_publish_rdbms_client import (
    OnlineStorePublishRdbmsClient,
)
from databricks.feature_store.online_store_publish_nosql_client import (
    OnlineStorePublishNoSqlClient,
)
from databricks.feature_store.online_store_spec import OnlineStoreSpec


def get_online_store_publish_client(
    online_store_spec: OnlineStoreSpec,
) -> OnlineStorePublishClient:
    if is_rdbms_spec(online_store_spec):
        return OnlineStorePublishRdbmsClient(online_store_spec)
    elif is_nosql_spec(online_store_spec):
        return OnlineStorePublishNoSqlClient(online_store_spec)
    raise ValueError(f"Unexpected online store type {type(online_store_spec)}")
