import uuid

import pyspark.sql.functions as f
from pyspark.sql import Window

# Publish latest feature values for tables with timestamp key.
# aka. feature values with the latest timestamp for each unique primary keys combination.
from databricks.feature_store.entities.online_store_detailed import OnlineStoreDetailed
from databricks.feature_store.online_store_spec import OnlineStoreSpec
from databricks.feature_store.online_store_publish_client import is_ttl_spec


def get_latest_snapshot(df, primary_keys, timestamp_keys):
    timestamp_key = timestamp_keys[0]
    # temporary timestamp rank key used for filtering, add random suffix to avoid column collision
    timestamp_rank_key = f"{timestamp_key}_rank_{uuid.uuid4().hex}"
    # partition DataFrame by primary keys and rank by timestamp key in descending order
    timestamp_rank_window = Window.partitionBy(primary_keys).orderBy(
        f.col(timestamp_key).desc()
    )
    # filter out the highest rank (latest snapshot) and drop the timestamp rank key
    return (
        df.withColumn(timestamp_rank_key, f.rank().over(timestamp_rank_window))
        .filter(f.col(timestamp_rank_key) == 1)
        .drop(timestamp_rank_key)
    )


def update_online_store_spec_sticky_ttl(
    online_store_spec: OnlineStoreSpec, online_store_detailed: OnlineStoreDetailed
):
    """
    Resolves the TTL and returns an updated OnlineStoreSpec. The OnlineStoreSpec TTL is used if defined.
    Otherwise, a sticky TTL is inherited from the OnlineStoreDetailed.
    """
    # Nothing needs to be done if the OnlineStoreSpec doesn't support TTL.
    if not is_ttl_spec(online_store_spec):
        return online_store_spec

    resolved_ttl = (
        online_store_spec.ttl
        if online_store_spec.ttl
        else online_store_detailed.additional_metadata.ttl
    )
    return online_store_spec.clone(**{"ttl": resolved_ttl})
