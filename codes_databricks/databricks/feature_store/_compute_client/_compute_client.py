import logging
from typing import List, Union, Dict, Any, Optional

import databricks.feature_store.utils.validation_utils
from databricks.feature_store.constants import (
    OVERWRITE,
    MERGE,
    DATA_TYPES_REQUIRES_DETAILS,
    _DEFAULT_WRITE_STREAM_TRIGGER,
    _WARN,
    _ERROR,
    _SOURCE_FORMAT_DELTA,
)
from databricks.feature_store.api.proto.feature_catalog_pb2 import ProducerAction
from databricks.feature_store.entities.data_type import DataType
from databricks.feature_store.entities.store_type import StoreType
from databricks.feature_store.entities.feature import Feature
from databricks.feature_store.entities.feature_table import FeatureTable

from databricks.feature_store.entities.key_spec import KeySpec
from databricks.feature_store.hive_client import HiveClient
from databricks.feature_store.catalog_client import CatalogClient
from databricks.feature_store.utils import utils
from databricks.feature_store.utils.spark_listener import SparkSourceListener
from databricks.feature_store.utils import request_context
from databricks.feature_store.utils.request_context import (
    RequestContext,
)
from databricks.feature_store.utils import schema_utils
from databricks.feature_store._hive_client_helper import HiveClientHelper
from databricks.feature_store._catalog_client_helper import CatalogClientHelper

from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import StructType
from pyspark.sql.streaming import StreamingQuery

_logger = logging.getLogger(__name__)


class ComputeClient:
    """
    The compute client manages metadata about feature tables, eg:

    - Creating/registering feature tables
    - Reading feature table metadata
    - Dropping feature tables from the catalog
    - Managing attributes of feature tables such as tags
    """

    _WRITE_MODES = [OVERWRITE, MERGE]

    def __init__(
        self,
        catalog_client: CatalogClient,
        catalog_client_helper: CatalogClientHelper,
        hive_client: HiveClient,
    ):
        self._catalog_client = catalog_client
        self._catalog_client_helper = catalog_client_helper
        self._hive_client = hive_client
        self._hive_client_helper = HiveClientHelper(self._hive_client)

    def create_table(
        self,
        name: str,
        primary_keys: Union[str, List[str]],
        df: Optional[DataFrame] = None,
        *,
        timestamp_keys: Union[str, List[str], None] = None,
        partition_columns: Union[str, List[str], None] = None,
        schema: Optional[StructType] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> FeatureTable:

        features_df = kwargs.pop("features_df", None)
        if features_df is not None and df is not None:
            raise ValueError("Either features_df or df can be provided, but not both.")
        if features_df is not None:
            _logger.warning(
                'The "features_df" parameter is deprecated. Use "df" instead.'
            )
            df = features_df
        path = kwargs.pop("path", None)
        databricks.feature_store.utils.validation_utils.check_kwargs_empty(
            kwargs, "create_table"
        )

        return self._create_table(
            name,
            primary_keys,
            df,
            timestamp_keys=timestamp_keys,
            partition_columns=partition_columns,
            schema=schema,
            description=description,
            path=path,
            tags=tags,
            req_context=RequestContext(request_context.CREATE_TABLE),
        )

    def _create_table(
        self,
        name: str,
        primary_keys: Union[str, List[str]],
        df: DataFrame = None,
        *,
        timestamp_keys: Union[str, List[str]] = None,
        partition_columns: Union[str, List[str]] = None,
        schema: StructType = None,
        description: str = None,
        path: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        req_context: Optional[RequestContext] = None,
    ) -> FeatureTable:

        if schema is None and df is None:
            raise ValueError("Either schema or df must be provided")

        if schema and df and not ComputeClient._schema_eq(schema, df.schema):
            raise ValueError(
                "Provide either 'schema' or 'df' arguments. If both arguments "
                "are provided, their schemas must match."
            )

        if df is not None:
            databricks.feature_store.utils.validation_utils.check_dataframe_type(df)
        self._hive_client_helper.check_catalog_database_exists(name)

        table_schema = schema or df.schema
        ComputeClient._check_schema_top_level_types_supported(table_schema)

        partition_cols_as_list = utils.as_list(partition_columns, default=[])
        if partition_columns:
            ComputeClient._check_schema_has_columns(
                table_schema, partition_cols_as_list, "partition columns"
            )

        primary_keys_as_list = utils.as_list(primary_keys)
        ComputeClient._check_schema_has_columns(
            table_schema, primary_keys_as_list, "primary keys"
        )

        timestamp_keys_as_list = utils.as_list(timestamp_keys, default=[])
        if timestamp_keys:
            ComputeClient._check_schema_has_columns(
                table_schema, timestamp_keys_as_list, "timestamp keys"
            )

        # 1. Handle cases where the table exists in either Hive or the Catalog
        delta_table_exists = self._hive_client.table_exists(name)
        catalog_table_exists = self._catalog_client.feature_table_exists(
            name, req_context
        )

        if delta_table_exists and not catalog_table_exists:
            raise ValueError(f"Data table {name} already exists. Use a different name.")

        if catalog_table_exists and not delta_table_exists:
            raise ValueError(
                f"Feature table {name} already exists, but data table not accessible in Spark. "
                f"Consider deleting the feature table to resolve this error."
            )

        if catalog_table_exists and delta_table_exists:
            return self._check_catalog_matches_delta_metadata(
                name,
                table_schema,
                primary_keys_as_list,
                partition_cols_as_list,
                timestamp_keys_as_list,
                req_context,
            )

        # At this point, neither the Delta table nor the Catalog table exist.

        # 2. Create empty Delta table. If this fails for some reason, the Feature Table will not be
        # added to the Feature Catalog.
        self._hive_client.create_table(name, table_schema, partition_cols_as_list, path)

        # 3. Add feature table and features to the Feature Catalog.
        # Features (other than primary keys and partition keys) are added in a separate call.
        delta_schema = {
            feature.name: feature.dataType
            for feature in self._hive_client.get_feature_table_schema(name)
        }

        # TODO(ML-17484): Move the following checks prior to Hive table creation

        partition_key_specs = []
        for k in partition_cols_as_list:
            spark_data_type = delta_schema[k]
            partition_key_specs.append(KeySpec(k, spark_data_type.typeName()))

        primary_key_specs = []
        for k in primary_keys_as_list:
            spark_data_type = delta_schema[k]
            primary_key_specs.append(KeySpec(k, spark_data_type.typeName()))

        timestamp_key_specs = []
        for k in timestamp_keys_as_list:
            spark_data_type = delta_schema[k]
            timestamp_key_specs.append(KeySpec(k, spark_data_type.typeName()))

        feature_key_specs = self._get_feature_key_specs(
            delta_schema,
            primary_keys_as_list,
            timestamp_keys_as_list,
            partition_cols_as_list,
        )

        try:
            self._create_feature_table_with_features_and_tags(
                name=name,
                partition_key_specs=partition_key_specs,
                primary_key_specs=primary_key_specs,
                timestamp_key_specs=timestamp_key_specs,
                description=description,
                is_imported=False,
                feature_key_specs=feature_key_specs,
                tags=tags,
                req_context=req_context,
            )
        except Exception as e:
            # Delete empty Delta table.  The feature table will have already been cleaned up from the catalog.
            self._hive_client.delete_empty_table(name)
            raise e

        # 4. Write to Delta table
        if df is not None:
            try:
                # Use mode OVERWRITE since this a new feature table.
                self.write_table(
                    name,
                    df,
                    mode=OVERWRITE,
                    producer_action=ProducerAction.CREATE,
                    req_context=req_context,
                )
            except Exception as e:
                # Delete the entire delta table if fatal exception occurs.
                # This may happen after partial data was written and an unknown exception is thrown.
                # It is OK to delete the feature table here because we are certain this is the
                # feature table we just created. We should NOT delete any existing feature table
                # created by user
                self._hive_client.delete_table(name)
                self._catalog_client.delete_feature_table(name, req_context)
                raise e

        _logger.info(f"Created feature table '{name}'.")
        return self.get_table(name, req_context)

    def register_table(
        self,
        *,
        delta_table: str,
        primary_keys: Union[str, List[str]],
        timestamp_keys: Union[str, List[str], None] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> FeatureTable:

        # Validate if the provided Delta table exists.
        req_context = RequestContext(request_context.REGISTER_TABLE)
        self._hive_client_helper.check_catalog_database_exists(delta_table)
        if not self._hive_client.table_exists(delta_table):
            raise ValueError(
                f"The provided Delta table '{delta_table}' could not be found."
            )
        df = self._hive_client.read_table(delta_table)

        # Validate if the provided Delta table is feature store compliant
        # 1. Check if the Delta table contains valid types and the specified primary key and
        #    timestamp key columns.
        table_schema = df.schema
        ComputeClient._check_schema_top_level_types_supported(table_schema)

        primary_keys_as_list = utils.as_list(primary_keys)
        ComputeClient._check_schema_has_columns(
            table_schema, primary_keys_as_list, "primary keys"
        )

        timestamp_keys_as_list = utils.as_list(timestamp_keys, default=[])
        if timestamp_keys:
            ComputeClient._check_schema_has_columns(
                table_schema, timestamp_keys_as_list, "timestamp keys"
            )

        # 2. Check that the Delta table does not contain generated columns.
        #    More details: go/import_delta_table
        if self._hive_client.has_generated_columns(delta_table):
            raise ValueError(
                f"Provided Delta table must not contain generated column(s)."
            )
        # 3. Handle the case where the Delta table is already a feature table.
        if self._catalog_client.feature_table_exists(delta_table, req_context):
            return self._check_catalog_matches_delta_metadata(
                delta_table,
                table_schema,
                primary_keys_as_list,
                [],
                timestamp_keys_as_list,
                req_context,
            )

        # 4. Check no two rows have the same primary keys.
        if self._hive_client.df_violates_pk_constraint(
            df, primary_keys_as_list + timestamp_keys_as_list
        ):
            raise ValueError(
                f"Non-unique rows detected in input dataframe for key combination"
                f"{ primary_keys_as_list + timestamp_keys_as_list}."
            )

        # Register the table with feature store
        delta_schema = {feature.name: feature.dataType for feature in table_schema}
        primary_key_specs = []
        for k in primary_keys_as_list:
            spark_data_type = delta_schema[k]
            primary_key_specs.append(KeySpec(k, spark_data_type.typeName()))

        timestamp_key_specs = []
        for k in timestamp_keys_as_list:
            spark_data_type = delta_schema[k]
            timestamp_key_specs.append(KeySpec(k, spark_data_type.typeName()))

        feature_key_specs = self._get_feature_key_specs(
            delta_schema,
            primary_keys_as_list,
            timestamp_keys_as_list,
            [],
        )

        self._create_feature_table_with_features_and_tags(
            name=delta_table,
            partition_key_specs=[],
            primary_key_specs=primary_key_specs,
            timestamp_key_specs=timestamp_key_specs,
            description=description,
            is_imported=True,
            feature_key_specs=feature_key_specs,
            tags=tags,
            req_context=req_context,
        )

        self._catalog_client_helper.add_producers(
            delta_table, ProducerAction.REGISTER, req_context
        )
        return self.get_table(delta_table, req_context)

    def write_table(
        self,
        name: str,
        df: DataFrame,
        mode: str,
        req_context: RequestContext,
        checkpoint_location: Union[str, None] = None,
        trigger: Dict[str, Any] = _DEFAULT_WRITE_STREAM_TRIGGER,
        producer_action: ProducerAction = None,
    ) -> Union[StreamingQuery, None]:
        """
        Write to a feature table.

        If the input :class:`DataFrame <pyspark.sql.DataFrame>` is streaming, will create a write stream.

        :param name: A feature table name of the form ``<database_name>.<table_name>``,
          for example ``dev.user_features``. Raises an exception if this feature table does not
          exist.
        :param df: Spark :class:`DataFrame <pyspark.sql.DataFrame>` with feature data. Raises an exception if the schema does not
          match that of the feature table.
        :param mode: Two supported write modes:

          * ``"overwrite"`` updates the whole table.

          * ``"merge"`` will upsert the rows in ``df`` into the feature table. If ``df`` contains
            columns not present in the feature table, these columns will be added as new features.

        :param checkpoint_location: Sets the Structured Streaming ``checkpointLocation`` option.
          By setting a ``checkpoint_location``, Spark Structured Streaming will store
          progress information and intermediate state, enabling recovery after failures.
          This parameter is only supported when the argument ``df`` is a streaming :class:`DataFrame <pyspark.sql.DataFrame>`.
        :param trigger: If ``df.isStreaming``, ``trigger`` defines the timing of stream data
          processing, the dictionary will be unpacked and passed to :meth:`DataStreamWriter.trigger <pyspark.sql.streaming.DataStreamWriter.trigger>`
          as arguments. For example, ``trigger={'once': True}`` will result in a call to
          ``DataStreamWriter.trigger(once=True)``.
        :return: If ``df.isStreaming``, returns a PySpark :class:`StreaminQuery <pyspark.sql.streaming.StreamingQuery>`, :obj:`None` otherwise.
        """
        databricks.feature_store.utils.validation_utils.check_dataframe_type(df)
        mode_string = mode.strip().lower()
        if mode_string not in self._WRITE_MODES:
            supported_modes_list = ", ".join([f"'{m}'" for m in self._WRITE_MODES])
            raise ValueError(
                f"Unsupported mode '{mode}'. Use one of {supported_modes_list}"
            )

        checkpoint_location = databricks.feature_store.utils.validation_utils.standardize_checkpoint_location(
            checkpoint_location
        )
        if checkpoint_location is not None and not df.isStreaming:
            _logger.warning("Ignoring checkpoint_location, since df is not streaming.")
            checkpoint_location = None

        ComputeClient._check_schema_top_level_types_supported(df.schema)
        self._hive_client_helper.check_feature_table_exists(name)
        feature_table = self._catalog_client.get_feature_table(name, req_context)

        # We know from the successful `get_feature_table call` above that the user has
        # at least read permission. Otherwise backend will throw RESOURCE_DOES_NOT_EXIST exception.
        # Since this is a write operation, we want to check whether the user has write permission
        # on the feature table prior to other operations.
        if not self._catalog_client.can_write_to_catalog(name, req_context):
            raise PermissionError(
                f"You do not have permission to write to feature table {name}."
            )

        # Validate: Internal state is consistent. Existing Delta schema should match Catalog schema.
        features = self._catalog_client.get_features(name, req_context)
        existing_schema = self._hive_client.get_feature_table_schema(name)
        if not schema_utils.catalog_matches_delta_schema(features, existing_schema):
            # If the existing Delta table does not match the Feature Catalog, the state is invalid
            # and we cannot write to the feature table. Error out.
            schema_utils.log_catalog_schema_not_match_delta_schema(
                features, existing_schema, level=_ERROR
            )

        # Validate: Provided DataFrame has key and partition columns
        features = self._catalog_client.get_features(name, req_context)
        if not schema_utils.catalog_matches_delta_schema(
            features, df.schema, column_filter=feature_table.primary_keys
        ):
            raise ValueError(
                f"The provided DataFrame must contain all specified primary key columns and have "
                f"the same type. Could not find key(s) '{feature_table.primary_keys}' with "
                f"correct types in schema {df.schema}."
            )
        if not schema_utils.catalog_matches_delta_schema(
            features, df.schema, column_filter=feature_table.partition_columns
        ):
            raise ValueError(
                f"The provided DataFrame must contain all specified partition columns and have "
                f"the same type. Could not find partition column(s) "
                f"'{feature_table.partition_columns}' with correct types in schema {df.schema}."
            )
        if not schema_utils.catalog_matches_delta_schema(
            features, df.schema, column_filter=feature_table.timestamp_keys
        ):
            raise ValueError(
                f"The provided DataFrame must contain the specified timestamp key column "
                f"and have the same type. Could not find key '{feature_table.timestamp_keys[0]}' "
                f"with correct types in schema {df.schema}."
            )

        # Invariant: We know from a validation check above that the Delta table schema matches the
        # Catalog schema.

        # Check for schema differences between the Catalog feature table and df's schema.
        if not schema_utils.catalog_matches_delta_schema(features, df.schema):
            # If this is a feature table with point-in-time lookup timestamp keys.
            # Validate: all existing table columns are present in the df.
            if feature_table.timestamp_keys:
                feature_names = [feature.name for feature in features]
                df_column_names = [c.name for c in df.schema]
                missing_column_names = list(set(feature_names) - set(df_column_names))
                if missing_column_names:
                    raise ValueError(
                        f"Feature table has a timestamp column. When calling write_table "
                        f"the provided DataFrame must contain all the feature columns. "
                        f"Could not find column(s) '{missing_column_names}'."
                    )
            # Attempt to update both the Delta table and Catalog schemas.
            # New columns will be added, column type mismatch will raise an error.
            ComputeClient._check_unique_case_insensitive_schema(features, df.schema)
            # First update the Delta schema. Spark will handle any type changes, and throw on
            # incompatible types.
            self._update_delta_features(name, df.schema)
            # Now update the Catalog using *the types in the Delta table*. We do not use the types
            # in `df` here so we can defer schema merging logic to Spark.
            delta_schema = self._hive_client.get_feature_table_schema(name)
            self._update_catalog_features_with_delta_schema(
                name, feature_table, features, delta_schema, req_context
            )

        # Exclude self-referential data sources. Feature table should exist.
        feature_table_data_source = self._hive_client.get_delta_table_path(name)
        # set(None) can produce exception, set default to empty list
        excluded_paths = set(utils.as_list(feature_table_data_source, default=[]))

        # Write data to Delta table
        with SparkSourceListener() as spark_source_listener:
            return_value = self._hive_client.write_table(
                name,
                feature_table.primary_keys,
                feature_table.timestamp_keys,
                df,
                mode_string,
                checkpoint_location,
                trigger,
            )
            subscribed_data_sources = spark_source_listener.get_data_sources()

        tables = set()
        paths = set()
        for fmt, sources in subscribed_data_sources.items():
            # filter out source that are an exact match to the excluded paths.
            # ToDo(mparkhe): Currently Spark listener will not return subdirs as data sources,
            #                but in future investigate a clean mechanism to deduplicate,
            #                and eliminate redundant subdirs reported as (Delta) data sources.
            #                eg: ["dbfs:/X.db/Y", "dbfs:/X.db/Y/_delta_log/checkpoint..."]
            valid_sources = list(
                filter(
                    lambda source: source not in excluded_paths,
                    sources,
                )
            )
            if len(valid_sources) > 0:
                # We rely on the spark listener to determine whether a data source is a delta table
                # for now. However, spark listener categorize delta table by looking up the
                # leaf node in spark query plan whereas we categorize delta table by whether or not
                # it would show up in the `Data` tab. Inconsistency could happen if user reads a
                # delta directory as delta table through `spark.read.format("delta")`,
                # we should store such data source as a path rather than a delta table.
                if fmt == _SOURCE_FORMAT_DELTA:
                    for path in valid_sources:
                        # Convert table-paths to "db_name.table_name".
                        # Note: If a table-path does not match the top level DBFS path
                        #       it is preserved as is.
                        converted_table = self._hive_client.convert_to_table_format(
                            path
                        )
                        if converted_table == path:
                            # Failed to convert table-path to "db_name.table_name",
                            # record data source as a path
                            paths.add(path)
                        else:
                            tables.add(converted_table)
                            # Exclude DBFS paths for all the table data sources
                            excluded_paths.add(path)
                else:
                    paths.update(valid_sources)

        # filter out paths match or are subdirectory (or files) under excluded paths
        # Example: if excluded_paths = ["dbfs:/path/to/database.db/table]
        #          also exclude sub-paths like "dbfs:/path/to/database.db/table/subpath"
        #          but do not exclude "dbfs:/path/to/database.db/tablesubdir"
        # ToDo(mparkhe): In future investigate a clean mechanism to eliminate subdirs
        #                of path sources, if returned by Spark listener.
        #                eg: ["dbfs:/X/Y", "dbfs:/X/Y/subdir"] => ["dbfs:/X/Y"]
        valid_paths = list(
            filter(
                lambda source: all(
                    [
                        source != excluded_path
                        and not source.startswith(utils.as_directory(excluded_path))
                        for excluded_path in excluded_paths
                    ]
                ),
                paths,
            )
        )

        # record data sources to feature catalog
        if len(tables) > 0 or len(valid_paths) > 0:
            self._catalog_client_helper.add_data_sources(
                name=name,
                tables=tables,
                paths=valid_paths,
                custom_sources=set(),  # No custom_sources in auto tracked data sources
                req_context=req_context,
            )
        # record producer to feature catalog
        self._catalog_client_helper.add_producers(name, producer_action, req_context)

        return return_value

    def get_table(self, name: str, req_context: RequestContext) -> FeatureTable:
        self._hive_client_helper.check_feature_table_exists(name)
        feature_table = self._catalog_client.get_feature_table(name, req_context)
        features = self._catalog_client.get_features(name, req_context)
        df = self._hive_client.read_table(name)
        if not schema_utils.catalog_matches_delta_schema(features, df.schema):
            schema_utils.log_catalog_schema_not_match_delta_schema(
                features, df.schema, level=_WARN
            )
        tag_entities = self._catalog_client.get_feature_table_tags(
            feature_table.table_id, req_context
        )
        feature_table._tags = {
            tag_entity.key: tag_entity.value for tag_entity in tag_entities
        }
        return feature_table

    def drop_table(self, name: str) -> None:
        req_context = RequestContext(request_context.DROP_TABLE)

        delta_table_exist = self._hive_client.table_exists(name)
        feature_table_exists = self._catalog_client.feature_table_exists(
            name, req_context
        )

        # Handle cases where catalog data does not exist.
        if not feature_table_exists and delta_table_exist:
            raise ValueError(
                f"Delta table '{name}' is not a feature table. Use spark API to drop the delta table. "
                f"For more information on Spark API, "
                f"see https://docs.databricks.com/sql/language-manual/sql-ref-syntax-ddl-drop-table.html."
            )
        if not feature_table_exists and not delta_table_exist:
            raise ValueError(f"Feature table '{name}' does not exist.")

        feature_table = self._catalog_client.get_feature_table(name, req_context)

        # Delete the feature table and underlying delta table.
        # First perform a dry-run deletion of catalog data as the backend validates the API call.
        try:
            self._catalog_client.delete_feature_table(name, req_context, dry_run=True)
        except Exception as e:
            _logger.error(f"Unable to delete the feature table due to {e}.")
            raise e
        self._hive_client.delete_table(name)
        try:
            self._catalog_client.delete_feature_table(name, req_context)
        except Exception as e:
            _logger.error(
                f"Failed to delete the feature table from Feature Catalog due to {e}."
                f" To fix this, re-run the 'drop_table' method."
            )
            raise e
        _logger.warning(
            "Deleting a feature table can lead to unexpected failures in upstream "
            "producers and downstream consumers (models, endpoints, and scheduled jobs)."
        )
        if feature_table.online_stores:
            ComputeClient._log_online_store_info(feature_table.online_stores)

    def read_table(self, name: str, **kwargs) -> DataFrame:
        as_of_delta_timestamp = kwargs.pop("as_of_delta_timestamp", None)
        databricks.feature_store.utils.validation_utils.check_kwargs_empty(
            kwargs, "read_table"
        )

        req_context = RequestContext(request_context.READ_TABLE)
        self._hive_client_helper.check_feature_table_exists(name)
        df = self._hive_client.read_table(name, as_of_delta_timestamp)
        features = self._catalog_client.get_features(name, req_context)
        if not schema_utils.catalog_matches_delta_schema(features, df.schema):
            schema_utils.log_catalog_schema_not_match_delta_schema(
                features, df.schema, level=_WARN
            )
        # Add consumer of each feature as final step
        consumer_feature_table_map = {name: [feature.name for feature in features]}
        self._catalog_client_helper.add_consumers(
            consumer_feature_table_map, req_context
        )
        return df

    def set_feature_table_tag(self, *, table_name: str, key: str, value: str) -> None:
        utils.validate_params_non_empty(locals(), ["table_name", "key", "value"])
        req_context = RequestContext(request_context.SET_FEATURE_TABLE_TAG)
        ft = self.get_table(table_name, req_context)
        self._catalog_client.set_feature_table_tags(
            ft.table_id, {key: value}, req_context
        )

    def delete_feature_table_tag(self, *, table_name: str, key: str) -> None:
        utils.validate_params_non_empty(locals(), ["table_name", "key"])
        req_context = RequestContext(request_context.DELETE_FEATURE_TABLE_TAG)
        ft = self.get_table(table_name, req_context)
        if key not in ft.tags:
            _logger.warning(
                f'The tag "{key}" for feature table "{table_name}" was not found, so the delete operation has been skipped.'
            )
        else:
            self._catalog_client.delete_feature_table_tags(
                ft.table_id, [key], req_context
            )

    def _check_catalog_matches_delta_metadata(
        self,
        name,
        table_schema,
        primary_keys_as_list,
        partition_cols_as_list,
        timestamp_keys_as_list,
        req_context,
    ) -> FeatureTable:
        """
        Checks if existing feature table catalog metadata with {name} matches the data table
        metadata including the table_schema, primary keys, timestamp keys and partition columns.
        Return the existing feature table if there is a match, otherwise raise an error.
        """
        ft = self.get_table(name, req_context)
        existing_features = self._catalog_client.get_features(name, req_context)
        schemas_match = schema_utils.catalog_matches_delta_schema(
            existing_features, table_schema
        )
        primary_keys_match = primary_keys_as_list == ft.primary_keys
        partition_keys_match = partition_cols_as_list == ft.partition_columns
        timestamp_keys_match = timestamp_keys_as_list == ft.timestamp_keys
        if (
            schemas_match
            and primary_keys_match
            and partition_keys_match
            and timestamp_keys_match
        ):
            _logger.warning(
                f'The feature table "{name}" already exists. Use "FeatureStoreClient.write_table"'
                f" API to write to the feature table."
            )
            return ft
        else:
            error_msg = (
                f"The feature table '{name}' already exists with a different schema.:\n"
            )
            if not schemas_match:
                error_msg += (
                    f"Existing schema: {existing_features}\n"
                    f"New schema:{table_schema}\n\n"
                )
            if not primary_keys_match:
                error_msg += (
                    f"Existing primary keys: {ft.primary_keys}\n"
                    f"New primary keys: {primary_keys_as_list}\n\n"
                )
            if not partition_keys_match:
                error_msg += (
                    f"Existing partition keys: {ft.partition_columns}\n"
                    f"New partition keys: {partition_cols_as_list}\n\n"
                )
            if not timestamp_keys_match:
                error_msg += (
                    f"Existing timestamp keys: {ft.timestamp_keys}\n"
                    f"New timestamp keys: {timestamp_keys_as_list}\n\n"
                )

            raise ValueError(error_msg)

    @staticmethod
    def _schema_eq(schema1, schema2):
        return set(schema1.fields) == set(schema2.fields)

    @staticmethod
    def _check_schema_top_level_types_supported(schema: StructType) -> None:
        """
        Checks whether the provided schema is supported by Feature Store, only considering the
        top-level type for nested data types.
        """
        unsupported_name_type = [
            (field.name, field.dataType)
            for field in schema.fields
            if not DataType.top_level_type_supported(field.dataType)
        ]
        if unsupported_name_type:
            plural = len(unsupported_name_type) > 1
            missing_cols_str = ", ".join(
                [
                    f"\n\t- {feat_name} (type: {feat_type})"
                    for (feat_name, feat_type) in unsupported_name_type
                ]
            )
            raise ValueError(
                f"Unsupported data type for column{'s' if plural else ''}: {missing_cols_str}"
            )

    @staticmethod
    def _check_schema_has_columns(schema, columns, col_type):
        schema_cols = [field.name for field in schema.fields]
        for col in columns:
            if col not in schema_cols:
                raise ValueError(
                    f"The provided DataFrame or schema must contain all specified {col_type}. "
                    f"Schema {schema} is missing column '{col}'"
                )

    @staticmethod
    def _get_feature_key_specs(
        delta_schema: StructType,
        primary_keys_as_list: List[str],
        timestamp_keys_as_list: List[str],
        partition_cols_as_list: List[str],
    ) -> List[KeySpec]:
        """
        Returns the KeySpec for only features in the delta_schema. KeySpecs are not created for
        primary keys, partition keys, and timestamp keys.
        """
        feature_key_specs = []
        for k in delta_schema:
            if (
                k not in partition_cols_as_list
                and k not in primary_keys_as_list
                and k not in timestamp_keys_as_list
            ):
                spark_data_type = delta_schema[k]
                # If the feature is a complex Spark DataType, convert the Spark DataType to its
                # JSON representation to be updated in the Feature Catalog.
                data_type_details = (
                    spark_data_type.json()
                    if DataType.from_spark_type(spark_data_type)
                    in DATA_TYPES_REQUIRES_DETAILS
                    else None
                )
                feature_key_specs.append(
                    KeySpec(k, spark_data_type.typeName(), data_type_details)
                )
        return feature_key_specs

    def _create_feature_table_with_features_and_tags(
        self,
        *,
        name: str,
        partition_key_specs: List[KeySpec],
        primary_key_specs: List[KeySpec],
        timestamp_key_specs: List[KeySpec],
        feature_key_specs: List[KeySpec],
        is_imported: bool,
        tags: Optional[Dict[str, str]] = None,
        description: str = None,
        req_context: Optional[RequestContext] = None,
    ) -> FeatureTable:
        """
        Create the feature_table, features and tags.

        If any step fails, the exception handler cleans up the feature table from the feature catalog
        and propagates the exception to the caller for further handling.
        """
        feature_table = None
        try:
            feature_table = self._catalog_client.create_feature_table(
                name,
                partition_key_spec=partition_key_specs,
                primary_key_spec=primary_key_specs,
                timestamp_key_spec=timestamp_key_specs,
                description=description,
                is_imported=is_imported,
                req_context=req_context,
            )
            if len(feature_key_specs) > 0:
                self._catalog_client.create_features(
                    name, feature_key_specs, req_context
                )
            if tags:
                self._catalog_client.set_feature_table_tags(
                    feature_table.table_id, tags, req_context
                )
            return feature_table
        except Exception as e:
            # Delete the newly created feature table in the catalog
            if feature_table:
                self._catalog_client.delete_feature_table(name, req_context)
            raise e

    # TODO(ML-21475): Move this logic to entities class
    @staticmethod
    def _log_online_store_info(online_stores):
        message = "You must delete the following published online stores with your cloud provider: \n"
        for online_store in online_stores:
            message += f"\t - '{online_store.name}' ({utils.get_canonical_online_store_name(online_store)}) \n"
            if online_store.store_type == StoreType.DYNAMODB:
                message += f"\t\t - Region: {online_store.dynamodb_metadata.region}, Table_arn: {online_store.dynamodb_metadata.table_arn} \n"
            elif online_store.store_type == StoreType.MYSQL:
                message += f"\t\t - Host: {online_store.mysql_metadata.host}, Port: {online_store.mysql_metadata.port} \n"
            elif online_store.store_type == StoreType.SQL_SERVER:
                message += f"\t\t - Host: {online_store.sql_server_metadata.host}, Port: {online_store.sql_server_metadata.port} \n"
            else:
                message += "Unknown online store."
        _logger.warning(message)

    @staticmethod
    def _check_unique_case_insensitive_schema(
        catalog_features: List[Feature], df_schema: DataFrame
    ) -> None:
        """
        Verify schema is unique and case sensitive.

        Confirm that column names in Feature Catalog and user's input
        DataFrame do not have duplicate
        case insensitive columns when writing data to the feature table.

        Prevents the following cases:
        1. User input DataFrame's schema is '{'feat1': 'FLOAT', 'FEAT1': 'FLOAT'}'
        2. User input DataFrame's schema is '{'FEAT1': 'FLOAT'}', and Feature Catalog's schema is
        '{'feat1': 'FLOAT'}'
        """
        df_cols = {}
        for df_column in df_schema:
            if df_column.name.lower() in df_cols:
                raise ValueError(
                    f"The provided DataFrame cannot contain duplicate column names. Column names are case insensitive. "
                    f"The DataFrame contains duplicate columns: {df_cols[df_column.name.lower()]}, {df_column.name}"
                )
            df_cols[df_column.name.lower()] = df_column.name

        for feature in catalog_features:
            if (
                feature.name.lower() in df_cols
                and feature.name != df_cols[feature.name.lower()]
            ):
                raise ValueError(
                    f"Feature names cannot differ by only case. The provided DataFrame has column "
                    f"{df_cols[feature.name.lower()]}, which duplicates the Feature Catalog column {feature.name}. "
                    f"Please rename the column"
                )

    def _update_delta_features(self, name, schema):
        """
        Update the Delta table with name `name`.

        This update happens by merging in `schema`. Will throw if the schema
        is incompatible with the existing Delta table schema.

        .. note::

           Validate: Delta table schemas are compatible. Because HiveClient.write_table enables
           the "mergeSchema" option, differences in schema will be reconciled by Spark. We will
           later write this schema to the Feature Catalog. In this way, we defer the schema
           merging logic to Spark.
        """
        try:
            self._hive_client.attempt_to_update_delta_table_schema(name, schema)
        except AnalysisException as e:
            raise ValueError(
                "FeatureStoreClient uses Delta APIs. The schema of the new DataFrame is "
                f"incompatible with existing Delta table. Saw AnalysisException: {str(e)}"
            )

    def _update_catalog_features_with_delta_schema(
        self, name, ft, features, delta_schema, req_context: RequestContext
    ):
        """
        Update the catalog to include all columns of the provided Delta table schema.

        :param name: Feature table name
        :param ft: FeatureTable
        :param features: [Features]
        :param delta_schema: Schema of the data table.
        :param req_context: The RequestContext
        """
        catalog_features_to_fs_types = {
            f.name: DataType.from_string(f.data_type) for f in features
        }
        delta_features_to_fs_types = {
            feature.name: DataType.from_spark_type(feature.dataType)
            for feature in delta_schema
        }
        complex_catalog_features_to_spark_types = (
            schema_utils.get_complex_catalog_schema(
                features, catalog_features_to_fs_types
            )
        )
        complex_delta_features_to_spark_types = schema_utils.get_complex_delta_schema(
            delta_schema, delta_features_to_fs_types
        )

        feaures_and_data_types_to_add = []
        features_and_data_types_to_update = []
        for feat, fs_data_type in delta_features_to_fs_types.items():
            simple_types_mismatch = (feat in catalog_features_to_fs_types) and (
                fs_data_type != catalog_features_to_fs_types[feat]
            )
            complex_types_mismatch = (
                feat in complex_catalog_features_to_spark_types
            ) and (
                complex_catalog_features_to_spark_types[feat]
                != complex_delta_features_to_spark_types[feat]
            )
            if simple_types_mismatch or complex_types_mismatch:
                # If the feature is a complex Spark DataType, convert the Spark DataType to its
                # JSON representation to be updated in the Feature Catalog.
                data_type_details = complex_delta_features_to_spark_types.get(feat)
                if data_type_details:
                    data_type_details = data_type_details.json()
                features_and_data_types_to_update.append(
                    (feat, fs_data_type, data_type_details)
                )
            if feat not in ft.primary_keys and feat not in catalog_features_to_fs_types:
                # If the feature is a complex Spark DataType, convert the Spark DataType to its
                # JSON representation to be updated in the Feature Catalog.
                data_type_details = complex_delta_features_to_spark_types.get(feat)
                if data_type_details:
                    data_type_details = data_type_details.json()
                feaures_and_data_types_to_add.append(
                    (feat, fs_data_type, data_type_details)
                )
        if feaures_and_data_types_to_add:
            key_specs = [
                KeySpec(
                    feat,
                    DataType.to_string(data_type),
                    data_type_details,
                )
                for (
                    feat,
                    data_type,
                    data_type_details,
                ) in feaures_and_data_types_to_add
            ]
            self._catalog_client.create_features(name, key_specs, req_context)
        # There is no need to update types of existing columns because mergeSchema does not support
        # column type changes.
