from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Optional

from databricks.feature_store.constants import (
    OVERWRITE,
    MERGE,
)
from databricks.feature_store.online_store_spec import (
    OnlineStoreSpec,
    AzureSqlServerSpec,
    AzureMySqlSpec,
    AmazonRdsMySqlSpec,
    AmazonDynamoDBSpec,
    AzureCosmosDBSpec,
)

RDBMS_SPECS = [AzureSqlServerSpec, AzureMySqlSpec, AmazonRdsMySqlSpec]
NOSQL_SPECS = [AmazonDynamoDBSpec, AzureCosmosDBSpec]
TTL_SPECS = [AmazonDynamoDBSpec]


def is_rdbms_spec(online_store_spec: OnlineStoreSpec):
    return type(online_store_spec) in RDBMS_SPECS


def is_nosql_spec(online_store_spec: OnlineStoreSpec):
    return type(online_store_spec) in NOSQL_SPECS


def is_ttl_spec(online_store_spec: OnlineStoreSpec):
    """
    Returns if the OnlineStoreSpec supports time to live.
    """
    return type(online_store_spec) in TTL_SPECS


class OnlineTable(ABC):
    def __init__(
        self, name: str, cloud_provider_unique_id: Optional[str], new_table: bool
    ):
        self._name = name
        self._cloud_provider_unique_id = cloud_provider_unique_id
        self._new_table = new_table

    @property
    def name(self):
        return self._name

    @property
    def cloud_provider_unique_id(self):
        return self._cloud_provider_unique_id

    @property
    def is_new_empty_table(self):
        return self._new_table


class OnlineStorePublishClient(ABC):
    def publish(
        self,
        df,
        primary_keys,
        timestamp_keys,
        streaming,
        mode,
        trigger,
        checkpoint_location=None,
        lookback_window: Optional[timedelta] = None,
    ):
        """Write the contents of df to the online store.

        The online store information is obtained from the OnlineStoreSpec
        passed to the derived class constructor. If the online store already contains a table with
        the specified table_name, new data will be merged if mode=MERGE or overwritten
        if mode=OVERWRITE.  mode=OVERWRITE is not supported when streaming=True.

        For more details, see the docstring for
        FeatureStoreClient.publish_table.

        Returns pyspark.sql.streaming.StreamingQuery if streaming=True, else None

        :param lookback_window: This parameter is extracted from the OnlineStoreSpec as part of
          FeatureStoreClient.publish_table, and is only applicable when publishing time series feature tables.
          It defines how far to lookback in the data event time (defined by the timestamp key) from the system time
           when publishing. For example, we publish all data where data event time > system time - lookback window.
        """
        if streaming:
            # The default trigger interval of 5 minutes is arbitrary. We may later run
            # load tests to see if processingTime can be decreased without causing high
            # load on the online store.
            # TODO: Scale test & solicit feedback from the structured streaming team.
            options = {}
            if checkpoint_location is not None:
                options["checkpointLocation"] = checkpoint_location

            # Streaming only supports merge mode, so we call self._publish_merge.
            return (
                df.writeStream.trigger(**trigger)
                .outputMode("update")
                .foreachBatch(
                    lambda mb, _: self._publish_merge(
                        mb, primary_keys, timestamp_keys, lookback_window
                    )
                )
                .options(**options)
                .start()
            )
        # TODO [ML-18325]: move get latest snapshot logic from functions below to client.py
        # Since spark streaming does not natively support get_latest_snapshot logic,
        # we are nesting this logic inside each micro-batch operation. Once this operation
        # is supported by Spark streaming natively in the future, we should extra this logic out
        # to client.py before calling PublishClient.publish().
        elif mode == MERGE:
            self._publish_merge(df, primary_keys, timestamp_keys, lookback_window)
        elif mode == OVERWRITE:
            self._publish_overwrite(df, primary_keys, timestamp_keys, lookback_window)
        else:
            # Should never get here since "mode" should be checked prior to calling
            # this internal API.
            raise ValueError("Invalid publish mode")
        self.close()

    @abstractmethod
    def get_or_create_online_table(self, df, primary_keys, timestamp_keys):
        """
        Create empty online table with expected schema if not exists.
        Returns OnlineTable
        """
        raise NotImplementedError

    @abstractmethod
    def _publish_overwrite(
        self,
        df,
        primary_keys,
        timestamp_keys,
        lookback_window: Optional[timedelta],
    ):
        raise NotImplementedError

    @abstractmethod
    def _publish_merge(
        self,
        df,
        primary_keys,
        timestamp_keys,
        lookback_window: Optional[timedelta],
    ):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
