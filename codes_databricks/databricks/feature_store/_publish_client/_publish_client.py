import databricks.feature_store.utils.validation_utils
from databricks.feature_store.catalog_client import CatalogClient
from databricks.feature_store.online_store_spec import OnlineStoreSpec
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST

from typing import List, Union, Dict, Any, Optional

from databricks.feature_store.constants import (
    OVERWRITE,
    MERGE,
    _DEFAULT_PUBLISH_STREAM_TRIGGER,
)
from databricks.feature_store.online_store_publish_client_factory import (
    get_online_store_publish_client,
)
from databricks.feature_store.utils import request_context
from databricks.feature_store.utils.publish_utils import (
    update_online_store_spec_sticky_ttl,
)
from databricks.feature_store.utils.request_context import (
    RequestContext,
    PUBLISH_AUTH_TYPE,
)
from databricks.feature_store.utils import utils
from pyspark.sql.streaming import StreamingQuery
import logging
from databricks.feature_store.online_store_publish_client import is_ttl_spec
from databricks.feature_store.entities.online_store_metadata import OnlineStoreMetadata
from databricks.feature_store.utils.rest_utils import get_error_code
from databricks.feature_store.hive_client import HiveClient
from databricks.feature_store.utils import schema_utils
from databricks.feature_store._hive_client_helper import HiveClientHelper

_logger = logging.getLogger(__name__)


class PublishClient:

    _PUBLISH_MODES = [OVERWRITE, MERGE]

    def __init__(
        self,
        catalog_client: CatalogClient,
        hive_client: HiveClient,
        hive_client_helper: HiveClientHelper,
    ):
        self._catalog_client = catalog_client
        self._hive_client = hive_client
        self._hive_client_helper = hive_client_helper

    def _publish_table(
        self,
        name: str,
        online_store: OnlineStoreSpec,
        *,
        filter_condition: Optional[str] = None,
        mode: str = MERGE,
        streaming: bool = False,
        checkpoint_location: Optional[str] = None,
        trigger: Dict[str, Any] = _DEFAULT_PUBLISH_STREAM_TRIGGER,
        features: Union[str, List[str], None] = None,
    ) -> Optional[StreamingQuery]:

        req_context = RequestContext(request_context.PUBLISH_TABLE)

        # Check: valid mode used
        mode_string = mode.strip().lower()
        if mode_string not in self._PUBLISH_MODES:
            supported_modes_list = ", ".join([f"'{m}'" for m in self._PUBLISH_MODES])
            raise ValueError(
                f"Unsupported mode '{mode}'. Use one of ({supported_modes_list})"
            )

        # Check: arguments and combination of arguments
        if streaming and mode == OVERWRITE:
            raise ValueError(
                f'Streaming publish_table does not support mode="{OVERWRITE}"'
            )

        # Empty input_features means the default behavior of publishing the entire table.
        input_features = utils.as_list(features, [])
        if input_features and mode == OVERWRITE:
            raise ValueError(
                f'publish_table for selected columns does not support mode="{OVERWRITE}"'
            )

        checkpoint_location = databricks.feature_store.utils.validation_utils.standardize_checkpoint_location(
            checkpoint_location
        )
        if checkpoint_location is not None and not streaming:
            _logger.warning("Ignoring checkpoint_location, since df is not streaming.")
            checkpoint_location = None

        # Check: feature table exists as a delta table
        self._hive_client_helper.check_feature_table_exists(name)

        # Check: feature tables exists in the catalog
        if not self._catalog_client.feature_table_exists(name, req_context):
            raise ValueError(f"Feature table '{name}' does not exists in the catalog.")

        catalog_feature_table = self._catalog_client.get_feature_table(
            name, req_context
        )

        # Check: schema between delta table and in catalog
        hive_feature_table_df = self._hive_client.read_table(name, None)
        catalog_features = self._catalog_client.get_features(name, req_context) or []

        # ToDo(mparkhe): DRY this code with schema comparison in write_tables
        # Check: Primary keys
        if not schema_utils.catalog_matches_delta_schema(
            catalog_features,
            hive_feature_table_df.schema,
            column_filter=catalog_feature_table.primary_keys or [],
        ):
            primary_keys = [
                pk
                for pk in catalog_features
                if pk.name in catalog_feature_table.primary_keys
            ]
            raise ValueError(
                f"Mismatched primary keys. Primary keys in the catalog, '{primary_keys}' do not "
                f"match keys in the data table schema {hive_feature_table_df.schema}."
            )

        # Check: Partition columns
        if (
            catalog_feature_table.partition_columns
            and not schema_utils.catalog_matches_delta_schema(
                catalog_features,
                hive_feature_table_df.schema,
                column_filter=catalog_feature_table.partition_columns or [],
            )
        ):
            partition_columns = [
                pk
                for pk in catalog_features
                if pk.name in catalog_feature_table.partition_columns
            ]
            raise ValueError(
                f"Mismatched partition columns. Partition columns  in the catalog, "
                f"'{partition_columns}' do not match keys in the data table "
                f"schema {hive_feature_table_df.schema}."
            )

        catalog_feature_names = {feature.name for feature in catalog_features}

        features_to_publish = None
        if input_features:
            input_features_names_lower_case = [
                feature_name.lower() for feature_name in input_features
            ]
            catalog_feature_names_lower_case = [
                feature_name.lower() for feature_name in catalog_feature_names
            ]
            input_features_not_in_catalog = [
                feature_name
                for feature_name in input_features_names_lower_case
                if feature_name not in catalog_feature_names_lower_case
            ]

            if len(input_features_not_in_catalog) > 0:
                raise ValueError(
                    f"Unknown feature column names. Input features, "
                    f"'{input_features_not_in_catalog}' not found in Feature Store."
                )
            input_features_correct_case = [
                feature_name
                for feature_name in catalog_feature_names
                if feature_name.lower() in input_features_names_lower_case
            ]
            # Add required primary key and timestamp key columns to user input features if they are not already included.
            features_to_publish = list(
                set(input_features_correct_case).union(
                    set(catalog_feature_table.keys),
                    set(catalog_feature_table.timestamp_keys),
                )
            )

        # Check: features listed in the catalog for publish are also available in the delta table
        catalog_feature_names_for_publish = (
            features_to_publish if features_to_publish else catalog_feature_names
        )
        features_for_publish_not_in_hive = [
            feature_name
            for feature_name in catalog_feature_names_for_publish
            if feature_name not in hive_feature_table_df.columns
        ]
        if len(features_for_publish_not_in_hive) > 0:
            raise RuntimeError(
                f"Schema mismatch. The following features being published were found in Feature Store, "
                f"but not in the data table: {sorted(features_for_publish_not_in_hive)}."
            )

        # check: features listed in the catalog not in the delta table and warn.
        # It warns when the selected columns we publish pass the previous check but there is a
        # mismatch for the unpublished columns.
        features_not_in_hive = [
            feature_name
            for feature_name in catalog_feature_names
            if feature_name not in hive_feature_table_df.columns
        ]
        if len(features_not_in_hive) > 0:
            _logger.warning(
                f"Schema mismatch. {features_not_in_hive} found in the feature catalog, "
                f"but were not found in the data table."
            )

        # Check: features not found catalog and warn.
        features_not_in_catalog = [
            feature_name
            for feature_name in hive_feature_table_df.columns
            if feature_name not in catalog_feature_names
        ]
        if len(features_not_in_catalog) > 0:
            if features_to_publish:
                _logger.warning(
                    f"Schema mismatch. {features_not_in_catalog} not found in the catalog. "
                )
            else:
                _logger.warning(
                    f"Schema mismatch. {features_not_in_catalog} not found in the catalog. "
                    f"Extra features in data table will not be published to the online store."
                )

        # Check: Time series publish attributes can only be used with time series feature tables.
        # This only needs to be checked prior to the first publish, as timestamp keys and ttl are immutable.
        # TTL is allowed to be None for both regular and time series (snapshot publish) feature tables.
        if is_ttl_spec(online_store) and online_store.ttl is not None:
            if len(catalog_feature_table.timestamp_keys) == 0:
                raise ValueError(
                    "Time to live can only be used with time series feature tables."
                )

        # Prepare DataFrame for publish
        publish_df = self._hive_client.read_table(
            name, as_of_delta_timestamp=None, streaming=streaming
        )

        # Filter rows
        if filter_condition:
            # force filtering before proceeding with other operations.
            # e.g. computing latest snapshot
            publish_df = publish_df.filter(filter_condition).cache()

        # Only publish features that exist in the catalog.
        if len(features_not_in_catalog) > 0:
            selected_columns = [
                c for c in publish_df.columns if c not in features_not_in_catalog
            ]
            publish_df = publish_df.select(*selected_columns)

        # Only publish features that are selected by user.
        if features_to_publish:
            user_selected_columns = [
                c for c in publish_df.columns if c in features_to_publish
            ]
            _logger.info(
                f"The following features will be published to the online store: [{user_selected_columns}]."
            )
            publish_df = publish_df.select(*user_selected_columns)

        # Augment the online store spec (e.g. resolve and store implicit values as part of the spec)
        online_store = online_store._augment_online_store_spec(name)

        # TODO (ML-22021): validate that there is only one timestamp key prior to publishing
        os_publish_client = get_online_store_publish_client(online_store)
        online_table = os_publish_client.get_or_create_online_table(
            publish_df,
            catalog_feature_table.primary_keys,
            catalog_feature_table.timestamp_keys,
        )

        # Update catalog metadata with publish log
        try:
            # Create the online store metadata object for use in publish
            online_store_metadata = OnlineStoreMetadata(
                online_store, online_table.cloud_provider_unique_id
            )

            # Call GetOnlineStore and retrieve the OnlineStoreDetailed from the Feature Catalog.
            # Then, update the online store spec with any relevant sticky values returned (e.g. ttl).
            # If no online store is found, nothing needs to be done since this is a new store.
            try:
                online_store_detailed = self._catalog_client.get_online_store(
                    name, online_store_metadata, req_context
                )
                online_store = update_online_store_spec_sticky_ttl(
                    online_store, online_store_detailed
                )
                # Update the online store metadata object to reflect the updates for the online store spec.
                online_store_metadata = OnlineStoreMetadata(
                    online_store, online_table.cloud_provider_unique_id
                )
            except Exception as e:
                # All exceptions not caused by a non-existent online store should be re-raised.
                if get_error_code(e) != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                    raise e

            publish_req_context = RequestContext(
                request_context.PUBLISH_TABLE,
                # Track the publish auth type.
                {PUBLISH_AUTH_TYPE: online_store.auth_type()},
            )
            # Resolve sticky values prior to calling PublishFeatureTable, as the `None` value is ambiguous.
            self._catalog_client.publish_feature_table(
                name,
                online_store_metadata,
                features_to_publish,
                publish_req_context,
            )
        except Exception as e:
            if online_table.is_new_empty_table:
                additional_msg = (
                    f"You may need to manually clean up the empty table '{online_table.name}' "
                    f"at the specified online store location. "
                )
            else:
                additional_msg = ""
            _logger.error(
                f"Failed to record online store in the catalog. {additional_msg}Exception: {e}",
                exc_info=True,
            )
            raise e

        # The lookback_window defaults to the online store TTL
        lookback_window = online_store.ttl if is_ttl_spec(online_store) else None

        # Create a new publish client, since the online store spec may have been updated.
        os_publish_client = get_online_store_publish_client(online_store)
        return os_publish_client.publish(
            publish_df,
            catalog_feature_table.primary_keys,
            catalog_feature_table.timestamp_keys,
            streaming,
            mode_string,
            trigger,
            checkpoint_location,
            lookback_window=lookback_window,
        )
