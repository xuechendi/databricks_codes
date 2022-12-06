import logging
from typing import List, Union, Dict, Any, Optional, Set
from types import ModuleType

from databricks.feature_store.api.proto.feature_catalog_pb2 import ProducerAction
from databricks.feature_store._compute_client._compute_client import ComputeClient
from databricks.feature_store._hive_client_helper import HiveClientHelper
from databricks.feature_store._catalog_client_helper import CatalogClientHelper
from databricks.feature_store.constants import (
    OVERWRITE,
    MERGE,
    CUSTOM,
    TABLE,
    PATH,
    _DEFAULT_WRITE_STREAM_TRIGGER,
    _DEFAULT_PUBLISH_STREAM_TRIGGER,
    _SOURCE_FORMAT_DELTA,
    _WARN,
    _ERROR,
)
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from databricks.feature_store.entities.feature import Feature
from databricks.feature_store.entities.feature_column_info import FeatureColumnInfo
from databricks.feature_store.entities.feature_table import FeatureTable
from databricks.feature_store.hive_client import HiveClient
from databricks.feature_store.catalog_client import CatalogClient
from databricks.feature_store.databricks_client import DatabricksClient
from databricks.feature_store.online_store_spec import (
    OnlineStoreSpec,
)
from databricks.feature_store.online_store_spec.online_store_properties import (
    HOSTNAME,
    PORT,
    DATABASE_NAME,
    TABLE_NAME,
)
from databricks.feature_store.training_set import TrainingSet
from databricks.feature_store.utils import utils
from databricks.feature_store.utils import request_context
from databricks.feature_store.utils.request_context import (
    RequestContext,
)
from databricks.feature_store.utils import validation_utils
from databricks.feature_store.utils import uc_utils
from databricks.feature_store._training_scoring_client._training_scoring_client import (
    TrainingScoringClient,
)
from databricks.feature_store._publish_client._publish_client import (
    PublishClient,
)

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from pyspark.sql.streaming import StreamingQuery

from mlflow.utils import databricks_utils
from mlflow.utils.annotations import deprecated, experimental
import mlflow

_logger = logging.getLogger(__name__)


class FeatureStoreClient:
    """
    Client for interacting with the Databricks Feature Store.
    """

    # !!!IMPORTANT!!!
    # All public facing method that has feature table name as input should get the full table name
    # based on the current catalog and schema set in the spark context.
    # See uc_utils.get_full_table_name() for details.

    _WRITE_MODES = [OVERWRITE, MERGE]
    _PUBLISH_MODES = [OVERWRITE, MERGE]
    _DATA_SOURCE_TYPES = [CUSTOM, PATH, TABLE]
    _DEFAULT_WRITE_STREAM_TRIGGER = _DEFAULT_WRITE_STREAM_TRIGGER
    _DEFAULT_PUBLISH_STREAM_TRIGGER = _DEFAULT_PUBLISH_STREAM_TRIGGER
    _SOURCE_FORMAT_DELTA = _SOURCE_FORMAT_DELTA
    _WARN = _WARN
    _ERROR = _ERROR

    def __init__(
        self,
        feature_store_uri: Optional[str] = None,
        model_registry_uri: Optional[str] = None,
    ):
        """
        Initialize a client to interact with the feature store.

        Creates a client to interact with the feature store. Takes in an optional parameter to identify the remote
        workspace for multi-workspace Feature Store.

        :param feature_store_uri: An URI of the form ``databricks://<scope>.<prefix>`` that identifies the credentials
          of the intended Feature Store workspace. Throws an error if specified but credentials were not found.
        :param model_registry_uri: Address of local or remote model registry server. If not provided,
          defaults to the local server.
        """

        if not utils.is_in_databricks_env():
            _logger.warning(
                f"The Databricks Feature Store client is intended to be run on Databricks. "
                f"Local and external development is not supported."
            )

        self._catalog_client = CatalogClient(
            databricks_utils.get_databricks_host_creds, feature_store_uri
        )
        # The Databricks client must be local from the context of the notebook
        self._databricks_client = DatabricksClient(
            databricks_utils.get_databricks_host_creds
        )
        self._hive_client = HiveClient()
        if not self._hive_client._on_spark_driver:
            _logger.warning(
                "Feature Store client functionality is limited when running outside of a Spark driver node. Spark operations will fail."
            )

        self._hive_client_helper = HiveClientHelper(self._hive_client)
        self._catalog_client_helper = CatalogClientHelper(
            self._catalog_client, self._databricks_client
        )
        self._compute_client = ComputeClient(
            catalog_client=self._catalog_client,
            catalog_client_helper=self._catalog_client_helper,
            hive_client=self._hive_client,
        )
        self._training_scoring_client = TrainingScoringClient(
            catalog_client=self._catalog_client,
            catalog_client_helper=self._catalog_client_helper,
            hive_client=self._hive_client,
            model_registry_uri=model_registry_uri,
        )
        self._publish_client = PublishClient(
            catalog_client=self._catalog_client,
            hive_client=self._hive_client,
            hive_client_helper=self._hive_client_helper,
        )
        self._model_registry_uri = model_registry_uri

    @deprecated(
        "FeatureStoreClient.create_table",
        since="v0.3.6",
    )
    def create_feature_table(
        self,
        name: str,
        keys: Union[str, List[str]],
        features_df: DataFrame = None,
        schema: StructType = None,
        partition_columns: Union[str, List[str]] = None,
        description: str = None,
        timestamp_keys: Union[str, List[str]] = None,
        **kwargs,
    ) -> FeatureTable:
        name = uc_utils.get_full_table_name(
            name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        path = kwargs.pop("path", None)
        validation_utils.check_kwargs_empty(kwargs, "create_feature_table")

        return self._compute_client._create_table(
            name,
            keys,
            features_df,
            schema=schema,
            partition_columns=partition_columns,
            description=description,
            timestamp_keys=timestamp_keys,
            path=path,
            req_context=RequestContext(request_context.CREATE_FEATURE_TABLE),
        )

    # TODO [ML-15539]: Replace the bitly URL in doc string
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
        """
        Create and return a feature table with the given name and primary keys.

        The returned feature table has the given name and primary keys.
        Uses the provided ``schema`` or the inferred schema
        of the provided ``df``. If ``df`` is provided, this data will be saved in
        a Delta table. Supported data types for features are: ``IntegerType``, ``LongType``,
        ``FloatType``, ``DoubleType``, ``StringType``, ``BooleanType``, ``DateType``,
        ``TimestampType``, ``ShortType``, ``ArrayType``, ``MapType``, and ``BinaryType``,
        and ``DecimalType``.

        :param name: A feature table name of the form ``<database_name>.<table_name>``,
          for example ``dev.user_features``.
        :param primary_keys: The feature table's primary keys. If multiple columns are required,
          specify a list of column names, for example ``['customer_id', 'region']``.
        :param df: Data to insert into this feature table. The schema of
          ``df`` will be used as the feature table schema.
        :param timestamp_keys: Columns containing the event time associated with feature value.
          Timestamp keys and primary keys of the feature table uniquely identify the feature value
          for an entity at a point in time.


          .. note::

             Experimental: This argument may change or be removed in
             a future release without warning.

        :param partition_columns: Columns used to partition the feature table. If a list is
          provided, column ordering in the list will be used for partitioning.

          .. Note:: When choosing partition columns for your feature table, use columns that do
                    not have a high cardinality. An ideal strategy would be such that you
                    expect data in each partition to be at least 1 GB.
                    The most commonly used partition column is a ``date``.

                    Additional info: `Choosing the right partition columns for Delta tables
                    <https://bit.ly/3ueXsjv>`_
        :param schema: Feature table schema. Either ``schema`` or ``df`` must be provided.
        :param description: Description of the feature table.
        :param tags: Tags to associate with the feature table.

          .. note::

            Available in version >= 0.4.1.

        :Other Parameters:
          * **path** (``Optional[str]``) --
            Path in a supported filesystem. Defaults to the database location.
        """
        name = uc_utils.get_full_table_name(
            name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        return self._compute_client.create_table(
            name=name,
            primary_keys=primary_keys,
            df=df,
            timestamp_keys=timestamp_keys,
            partition_columns=partition_columns,
            schema=schema,
            description=description,
            tags=tags,
            **kwargs,
        )

    def register_table(
        self,
        *,
        delta_table: str,
        primary_keys: Union[str, List[str]],
        timestamp_keys: Union[str, List[str], None] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> FeatureTable:
        """
        Register an existing Delta table as a feature table with the given primary keys.

        The returned feature table has the same name as the Delta table.

        .. note::

          Available in version >= 0.3.8.

        :param name: A Delta table name of the form ``<database_name>.<table_name>``,
          for example ``dev.user_features``. The table must exist in the metastore.
        :param primary_keys: The Delta table's primary keys. If multiple columns are required,
          specify a list of column names, for example ``['customer_id', 'region']``.
        :param timestamp_keys: Columns containing the event time associated with feature value.
          Together, the timestamp keys and primary keys uniquely identify the feature value at a point in time.
        :param description: Description of the feature table.
        :param tags: Tags to associate with the feature table.

          .. note::

            Available in version >= 0.4.1.

        :return: A :class:`FeatureTable <databricks.feature_store.entities.feature_table.FeatureTable>` object.
        """
        delta_table = uc_utils.get_full_table_name(
            delta_table,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        return self._compute_client.register_table(
            delta_table=delta_table,
            primary_keys=primary_keys,
            timestamp_keys=timestamp_keys,
            description=description,
            tags=tags,
        )

    def get_table(self, name: str) -> FeatureTable:
        """
        Get a feature table's metadata.

        :param name: A feature table name of the form ``<database_name>.<table_name>``, for
          example ``dev.user_features``.
        """
        name = uc_utils.get_full_table_name(
            name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        return self._compute_client.get_table(
            name=name, req_context=RequestContext(request_context.GET_TABLE)
        )

    @experimental
    def drop_table(self, name: str) -> None:
        """
        Delete the specified feature table. This API also drops the underlying Delta table.

        .. note::

          Available in version >= 0.4.1.

        :param name: The feature table name of the form ``<database_name>.<table_name>``,
            for example ``dev.user_features``.

        .. note::
            Deleting a feature table can lead to unexpected failures in  upstream producers and
            downstream consumers (models, endpoints, and scheduled jobs). You must delete any existing
            published online stores separately.
        """
        name = uc_utils.get_full_table_name(
            name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        self._compute_client.drop_table(name=name)

    @deprecated(
        "FeatureStoreClient.get_table",
        since="v0.3.6",
    )
    def get_feature_table(self, name: str) -> FeatureTable:
        name = uc_utils.get_full_table_name(
            name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        return self._compute_client.get_table(
            name=name, req_context=RequestContext(request_context.GET_FEATURE_TABLE)
        )

    def read_table(self, name: str, **kwargs) -> DataFrame:
        """
        Read the contents of a feature table.

        :param name: A feature table name of the form ``<database_name>.<table_name>``, for
          example ``dev.user_features``.
        :return: The feature table contents, or an exception will be raised if this feature table does not
          exist.
        """
        name = uc_utils.get_full_table_name(
            name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        return self._compute_client.read_table(name=name, **kwargs)

    def _get_feature_names_for_tables(
        self, req_context: RequestContext, table_names: Set[str]
    ) -> Dict[str, List[Feature]]:
        """
        Lookup features from the feature catalog for all table_names, return a dictionary of tablename -> list of features.
        """
        return {
            table_name: self._catalog_client.get_features(table_name, req_context)
            for table_name in table_names
        }

    def _get_feature_table_metadata_for_tables(
        self, req_context: RequestContext, table_names: Set[str]
    ) -> Dict[str, FeatureTable]:
        """
        Lookup FeatureTable metadata from the feature catalog for all table_names, return a dictionary of tablename -> FeatureTable.
        """
        return {
            table_name: self._catalog_client.get_feature_table(table_name, req_context)
            for table_name in table_names
        }

    def _load_feature_data_for_tables(
        self, table_names: Set[str]
    ) -> Dict[str, DataFrame]:
        """
        Load feature DataFrame objects for all table_names, return a dictionary of tablename -> DataFrame.
        """
        return {
            table_name: self._hive_client.read_table(table_name)
            for table_name in table_names
        }

    def _explode_feature_lookups(
        self,
        feature_lookups: List[FeatureLookup],
        feature_table_features_map: Dict[str, List[Feature]],
        feature_table_metadata_map: Dict[str, FeatureTable],
    ) -> List[FeatureColumnInfo]:
        """
        Explode FeatureLookups and collect into FeatureColumnInfos.  A FeatureLookup may explode into either:

        1. A single FeatureColumnInfo, in the case where only a single feature name is specified.
        2. Multiple FeatureColumnInfos, in the cases where either multiple or all feature names are specified.

        Additionally, when all feature names are specified in a FeatureLookup via setting feature_names to None,
        FeatureColumnInfos will be created for all features except primary keys.
        The order of the FeatureColumnInfos returned by this method will be the same order as returned by
        the backend:

        - All partition keys that are not primary keys, in the partition key order
        - All other non-key features
        """
        feature_column_infos = []
        for feature_lookup in feature_lookups:
            feature_column_infos_for_feature_lookup = self._explode_feature_lookup(
                feature_lookup=feature_lookup,
                features=feature_table_features_map[feature_lookup.table_name],
                feature_table=feature_table_metadata_map[feature_lookup.table_name],
            )
            feature_column_infos += feature_column_infos_for_feature_lookup
        return feature_column_infos

    def _explode_feature_lookup(
        self,
        feature_lookup: FeatureLookup,
        features: List[Feature],
        feature_table: FeatureTable,
    ) -> List[FeatureColumnInfo]:
        feature_names = []
        if feature_lookup._get_feature_names():
            # If the user explicitly passed in a feature name or list of feature names, use that
            feature_names += feature_lookup._get_feature_names()
        else:
            # Otherwise assume the user wants all columns in the feature table
            feature_names += [
                feature.name
                for feature in features
                # Filter out primary keys and timestamp keys
                if (
                    feature.name
                    not in [*feature_table.primary_keys, *feature_table.timestamp_keys]
                )
            ]

        return [
            FeatureColumnInfo(
                table_name=feature_lookup.table_name,
                feature_name=feature_name,
                lookup_key=utils.as_list(feature_lookup.lookup_key),
                output_name=(feature_lookup._get_output_name(feature_name)),
                timestamp_lookup_key=utils.as_list(
                    feature_lookup.timestamp_lookup_key, default=[]
                ),
            )
            for feature_name in feature_names
        ]

    def write_table(
        self,
        name: str,
        df: DataFrame,
        mode: str = MERGE,
        checkpoint_location: Optional[str] = None,
        trigger: Dict[str, Any] = _DEFAULT_WRITE_STREAM_TRIGGER,
    ) -> Optional[StreamingQuery]:
        """
        Writes to a feature table.

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
        :return: If ``df.isStreaming``, returns a PySpark :class:`StreamingQuery <pyspark.sql.streaming.StreamingQuery>`. :obj:`None` otherwise.
        """
        name = uc_utils.get_full_table_name(
            name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        return self._compute_client.write_table(
            name=name,
            df=df,
            mode=mode,
            req_context=RequestContext(request_context.WRITE_TABLE),
            checkpoint_location=checkpoint_location,
            trigger=trigger,
            producer_action=ProducerAction.WRITE,
        )

    @staticmethod
    def _index_of_online_store(
        online_store: OnlineStoreSpec, online_stores_list: List[Dict[str, str]]
    ):
        """
        TODO (ML-21692): Do not use this function until ML-21692 is resolved.

        Get the index of the online store.

        Returns the index of the provided online_store in the online_stores_list, or
        None.

        Two online stores are considered the same if they have the same
        database_name, host, and port.

        :param online_store: If the online_store is in the online_stores_list, its
          index will be returned.
        :param online_stores_list: A list of dictionaries of the properties of each
          online store.
        """

        def store_identifier(store_properties):
            return (
                store_properties[DATABASE_NAME],
                store_properties[TABLE_NAME],
                store_properties[HOSTNAME],
                store_properties[PORT],
            )

        online_store_identifier = store_identifier(online_store.non_secret_properties())
        online_store_to_index = {
            store_identifier(store): i for i, store in enumerate(online_stores_list)
        }
        return online_store_to_index.get(online_store_identifier)

    @experimental
    def add_data_sources(
        self,
        *,
        feature_table_name: str,
        source_names: Union[str, List[str]],
        source_type: str = "custom",
    ) -> None:
        """
        Add data sources to the feature table.

        :param feature_table_name: The feature table name.
        :param source_names: Data source names. For multiple sources,
            specify a list. If a data source name already exists, it is ignored.
        :param source_type: One of the following:

            * ``"table"``: Table in format <database_name>.<table_name> and is stored in the metastore (eg Hive).

            * ``"path"``: Path, eg in the Databricks File System (DBFS).

            * ``"custom"``: Manually added data source, neither a table nor a path.
        """
        feature_table_name = uc_utils.get_full_table_name(
            feature_table_name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        req_context = RequestContext(request_context.ADD_DATA_SOURCES)
        source_type_string = source_type.strip().lower()
        if source_type_string not in self._DATA_SOURCE_TYPES:
            supported_source_type_list = ", ".join(
                [f"'{m}'" for m in self._DATA_SOURCE_TYPES]
            )
            raise ValueError(
                f"Unsupported source_type '{source_type_string}'. Use one of ({supported_source_type_list})"
            )
        feature_table = self._catalog_client.get_feature_table(
            feature_table_name, req_context
        )
        if source_type_string == PATH:
            existing_sources_of_same_type = feature_table.path_data_sources
        elif source_type_string == TABLE:
            existing_sources_of_same_type = feature_table.table_data_sources
        elif source_type_string == CUSTOM:
            existing_sources_of_same_type = feature_table.custom_data_sources

        source_names_as_list = utils.as_list(source_names)
        duplicate_sources = [
            source
            for source in source_names_as_list
            if source in existing_sources_of_same_type
        ]
        if duplicate_sources:
            _logger.info(
                f"The following data source of type {source_type_string} already exists and will not be added: {duplicate_sources}."
            )

        new_sources = [
            source
            for source in source_names_as_list
            if source not in existing_sources_of_same_type
        ]
        if new_sources:
            tables = set(new_sources) if source_type_string == TABLE else set()
            paths = set(new_sources) if source_type_string == PATH else set()
            custom_sources = set(new_sources) if source_type_string == CUSTOM else set()
            self._catalog_client.add_data_sources(
                feature_table_name,
                tables=list(tables),
                paths=list(paths),
                custom_sources=list(custom_sources),
                req_context=req_context,
            )

    @experimental
    def delete_data_sources(
        self,
        *,
        feature_table_name: str,
        source_names: Union[str, List[str]],
    ) -> None:
        """
        Delete data sources from the feature table.

        .. Note:: Data sources of all types (table, path, custom) that match the source names will be deleted.

        :param feature_table_name: The feature table name.
        :param source_names: Data source names. For multiple sources,
            specify a list. If a data source name does not exist,
            it is ignored.

        """
        feature_table_name = uc_utils.get_full_table_name(
            feature_table_name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        req_context = RequestContext(request_context.DELETE_DATA_SOURCES)
        feature_table = self._catalog_client.get_feature_table(
            feature_table_name, req_context
        )
        existing_data_sources = (
            feature_table.path_data_sources
            + feature_table.table_data_sources
            + feature_table.custom_data_sources
        )
        source_names_as_list = utils.as_list(source_names)
        # initialize as a set to remove duplicate input source
        existing_data_sources_to_delete = {
            source for source in source_names_as_list if source in existing_data_sources
        }
        invalid_data_sources = [
            source
            for source in source_names_as_list
            if source not in existing_data_sources
        ]
        if invalid_data_sources:
            _logger.info(
                f"The following data sources do not exist and will not be deleted: {invalid_data_sources}."
            )
        if existing_data_sources_to_delete:
            self._catalog_client.delete_data_sources(
                feature_table_name,
                list(existing_data_sources_to_delete),
                req_context,
            )

    def publish_table(
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
        """
        Publish a feature table to an online store.

        :param name: Name of the feature table.
        :param online_store: Specification of the online store.
        :param filter_condition: A SQL expression using feature table columns that filters feature
          rows prior to publishing to the online store. For example, ``"dt > '2020-09-10'"``. This
          is analogous to running ``df.filter`` or a ``WHERE`` condition in SQL on a feature table
          prior to publishing.
        :param mode: Specifies the behavior when data already exists in this feature
          table in the online store. If ``"overwrite"`` mode is used, existing data is
          replaced by the new data. If ``"merge"`` mode is used, the new data will be
          merged in, under these conditions:

          * If a key exists in the online table but not the offline table,
            the row in the online table is unmodified.

          * If a key exists in the offline table but not the online table,
            the offline table row is inserted into the online table.

          * If a key exists in both the offline and the online tables,
            the online table row will be updated.

        :param streaming: If ``True``, streams data to the online store.
        :param checkpoint_location: Sets the Structured Streaming ``checkpointLocation`` option.
          By setting a ``checkpoint_location``, Spark Structured Streaming will store
          progress information and intermediate state, enabling recovery after failures.
          This parameter is only supported when ``streaming=True``.
        :param trigger: If ``streaming=True``, ``trigger`` defines the timing of
          stream data processing. The dictionary will be unpacked and passed
          to :meth:`DataStreamWriter.trigger <pyspark.sql.streaming.DataStreamWriter.trigger>` as arguments. For example, ``trigger={'once': True}``
          will result in a call to ``DataStreamWriter.trigger(once=True)``.
        :param features: Specifies the feature column(s) to be published to the online store.
          The selected features must be a superset of existing online store features. Primary key columns
          and timestamp key columns will always be published.

          .. Note:: This parameter is only supported when ``mode="merge"``. When ``features`` is not set, the whole feature table will be published.

        :return: If ``streaming=True``, returns a PySpark :class:`StreamingQuery <pyspark.sql.streaming.StreamingQuery>`, :obj:`None` otherwise.
        """
        name = uc_utils.get_full_table_name(
            name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        return self._publish_client._publish_table(
            name=name,
            online_store=online_store,
            filter_condition=filter_condition,
            mode=mode,
            streaming=streaming,
            checkpoint_location=checkpoint_location,
            trigger=trigger,
            features=features,
        )

    def create_training_set(
        self,
        df: DataFrame,
        feature_lookups: List[FeatureLookup],
        label: Union[str, List[str], None],
        exclude_columns: List[str] = [],
    ) -> TrainingSet:
        """
        Create a :class:`TrainingSet <databricks.feature_store.training_set.TrainingSet>`.

        :param df: The :class:`DataFrame <pyspark.sql.DataFrame>` used to join features into.
        :param feature_lookups: List of features to join into the :class:`DataFrame <pyspark.sql.DataFrame>`.
        :param label: Names of column(s) in :class:`DataFrame <pyspark.sql.DataFrame>` that contain training set labels. To create a training set without a label field, i.e. for unsupervised training set, specify label = None.
        :param exclude_columns: Names of the columns to drop from the :class:`TrainingSet <databricks.feature_store.training_set.TrainingSet>` :class:`DataFrame <pyspark.sql.DataFrame>`.
        :return: A :class:`TrainingSet <databricks.feature_store.training_set.TrainingSet>` object.
        """
        feature_lookups = uc_utils.get_feature_lookups_with_full_table_names(
            feature_lookups,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        return self._training_scoring_client.create_training_set(
            df=df,
            feature_lookups=feature_lookups,
            label=label,
            exclude_columns=exclude_columns,
        )

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        *,
        flavor: ModuleType,
        training_set: Optional[TrainingSet] = None,
        registered_model_name: Optional[str] = None,
        await_registration_for: int = mlflow.tracking._model_registry.DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        **kwargs,
    ):
        """
        Log an MLflow model packaged with feature lookup information.

        .. note::

           The :class:`DataFrame <pyspark.sql.DataFrame>` returned
           by :meth:`.TrainingSet.load_df` **must** be used to train the
           model. If it has been modified (for example data normalization, add a column,
           and similar), these modifications will not be applied at inference time,
           leading to training-serving skew.

        .. todo::

           [ML-15539]: Replace the bitly URL in doc string


        :param model: Model to be saved. This model must be capable of being saved by
          ``flavor.save_model``. See the `MLflow Model API
          <https://bit.ly/3yzl1r0>`_.
        :param artifact_path: Run-relative artifact path.
        :param flavor: MLflow module to use to log the model. ``flavor`` should have
          type :obj:`ModuleType <types.ModuleType>`.
          The module must have a method ``save_model``, and must support the ``python_function``
          flavor. For example, :mod:`mlflow.sklearn`, :mod:`mlflow.xgboost`, and similar.
        :param training_set: The :class:`.TrainingSet` used to train this model.
        :param registered_model_name:

          .. note::

             Experimental: This argument may change or be removed in
             a future release without warning.

          If given, create a model version under ``registered_model_name``,
          also creating a registered model if one with the given name does not exist.
        :param await_registration_for: Number of seconds to wait for the model version to finish
          being created and is in ``READY`` status. By default, the function waits for five minutes.
          Specify ``0`` or :obj:`None` to skip waiting.
        :return: `None`
        """
        self._training_scoring_client.log_model(
            model=model,
            artifact_path=artifact_path,
            flavor=flavor,
            training_set=training_set,
            registered_model_name=registered_model_name,
            await_registration_for=await_registration_for,
            **kwargs,
        )

    def score_batch(
        self, model_uri: str, df: DataFrame, result_type: str = "double"
    ) -> DataFrame:
        """
        Evaluate the model on the provided :class:`DataFrame <pyspark.sql.DataFrame>`.

        Additional features required for
        model evaluation will be automatically retrieved from :mod:`Feature Store <databricks.feature_store.client>`.

        .. todo::

           [ML-15539]: Replace the bitly URL in doc string

        The model must have been logged with :meth:`.FeatureStoreClient.log_model`,
        which packages the model with feature metadata. Unless present in ``df``,
        these features will be looked up from :mod:`Feature Store <databricks.feature_store.client>` and joined with ``df``
        prior to scoring the model.

        If a feature is included in ``df``, the provided feature values will be used rather
        than those stored in :mod:`Feature Store <databricks.feature_store.client>`.

        For example, if a model is trained on two features ``account_creation_date`` and
        ``num_lifetime_purchases``, as in:

        .. code-block:: python

            feature_lookups = [
                FeatureLookup(
                    table_name = 'trust_and_safety.customer_features',
                    feature_name = 'account_creation_date',
                    lookup_key = 'customer_id',
                ),
                FeatureLookup(
                    table_name = 'trust_and_safety.customer_features',
                    feature_name = 'num_lifetime_purchases',
                    lookup_key = 'customer_id'
                ),
            ]

            with mlflow.start_run():
                training_set = fs.create_training_set(
                    df,
                    feature_lookups = feature_lookups,
                    label = 'is_banned',
                    exclude_columns = ['customer_id']
                )
                ...
                  fs.log_model(
                    model,
                    "model",
                    flavor=mlflow.sklearn,
                    training_set=training_set,
                    registered_model_name="example_model"
                  )

        Then at inference time, the caller of :meth:`FeatureStoreClient.score_batch` must pass
        a :class:`DataFrame <pyspark.sql.DataFrame>` that includes ``customer_id``, the ``lookup_key`` specified in the
        ``FeatureLookups`` of the :mod:`training_set <databricks.feature_store.training_set>`.
        If the :class:`DataFrame <pyspark.sql.DataFrame>` contains a column
        ``account_creation_date``, the values of this column will be used
        in lieu of those in :mod:`Feature Store <databricks.feature_store.client>`. As in:

        .. code-block:: python

            # batch_df has columns ['customer_id', 'account_creation_date']
            predictions = fs.score_batch(
                'models:/example_model/1',
                batch_df
            )

        :param model_uri: The location, in URI format, of the MLflow model logged using
          :meth:`FeatureStoreClient.log_model`. One of:

            * ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            * ``models:/<model_name>/<model_version>``

            * ``models:/<model_name>/<stage>``

          For more information about URI schemes, see
          `Referencing Artifacts <https://bit.ly/3wnrseE>`_.
        :param df: The :class:`DataFrame <pyspark.sql.DataFrame>` to score the model on. :mod:`Feature Store <databricks.feature_store.client>` features will be joined with
          ``df`` prior to scoring the model. ``df`` must:

              1. Contain columns for lookup keys required to join feature data from Feature
              Store, as specified in the ``feature_spec.yaml`` artifact.

              2. Contain columns for all source keys required to score the model, as specified in
              the ``feature_spec.yaml`` artifact.

              3. Not contain a column ``prediction``, which is reserved for the model's predictions.
              ``df`` may contain additional columns.

        :param result_type: The return type of the model.
           See :func:`mlflow.pyfunc.spark_udf` result_type.
        :return: A :class:`DataFrame <pyspark.sql.DataFrame>`
           containing:

            1. All columns of ``df``.

            2. All feature values retrieved from Feature Store.

            3. A column ``prediction`` containing the output of the model.

        """
        return self._training_scoring_client.score_batch(
            model_uri=model_uri, df=df, result_type=result_type
        )

    def set_feature_table_tag(self, *, table_name: str, key: str, value: str) -> None:
        """Create or update a tag associated with the feature table. If the tag with the
        corresponding key already exists, its value will be overwritten with the new value.

        .. note::

          Available in version >= 0.4.1.

        :param table_name: the feature table name
        :param key: tag key
        :param value: tag value
        """
        table_name = uc_utils.get_full_table_name(
            table_name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        self._compute_client.set_feature_table_tag(
            table_name=table_name, key=key, value=value
        )

    def delete_feature_table_tag(self, *, table_name: str, key: str) -> None:
        """Delete the tag associated with the feature table. Deleting a non-existent tag will emit a warning.

        .. note::

          Available in version >= 0.4.1.

        :param table_name: the feature table name.
        :param key: the tag key to delete.
        """
        table_name = uc_utils.get_full_table_name(
            table_name,
            self._hive_client.get_current_catalog(),
            self._hive_client.get_current_database(),
        )
        self._compute_client.delete_feature_table_tag(table_name=table_name, key=key)
