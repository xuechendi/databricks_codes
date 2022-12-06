import logging

from typing import List, Dict, Optional

from databricks.feature_store.entities.consumer import Consumer
from databricks.feature_store.entities.feature_table import FeatureTable
from databricks.feature_store.entities.feature import Feature
from databricks.feature_store.entities.tag import Tag
from databricks.feature_store.entities.online_store_detailed import OnlineStoreDetailed
from databricks.feature_store.entities.online_store_metadata import OnlineStoreMetadata
from databricks.feature_store.entities._permission_level import _PermissionLevel
from databricks.feature_store.entities.online_feature_table import OnlineFeatureTable
from databricks.feature_store.utils.request_context import RequestContext
from databricks.feature_store.entities.store_type import StoreType

from databricks.feature_store.api.proto.feature_catalog_pb2 import (
    FeatureStoreService,
    CreateFeatureTable,
    GetFeatureTable,
    PublishFeatureTable,
    CreateFeatures,
    GetFeatures,
    UpdateFeature,
    KeySpec,
    DeleteFeatureTable,
    AddDataSources,
    DeleteDataSources,
    AddProducer,
    AddConsumer,
    GetConsumers,
    ConsumedFeatures,
    Notebook,
    Job,
    FeatureTableFeatures,
    GetModelServingMetadata,
    ProducerAction,
    Tag as ProtoTag,
    SetTags,
    DeleteTags,
    GetTags,
    GetOnlineStore,
)

from databricks.feature_store.utils.rest_utils import (
    call_endpoint,
    extract_api_info_for_service,
    _REST_API_PATH_PREFIX,
    proto_to_json,
    get_error_code,
)
from databricks.feature_store.utils.uc_utils import reformat_full_table_name
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST
from mlflow.utils import databricks_utils

_METHOD_TO_INFO = extract_api_info_for_service(
    FeatureStoreService, _REST_API_PATH_PREFIX
)

_logger = logging.getLogger(__name__)


class CatalogClient:
    """
    This provides the client interface to the backend feature catalog service running in the Databricks Control Plane.

    The catalog client should be reserved for low-level catalog operations and not contain any business logic
    that is unrelated to the catalog itself (for example, calling other Databricks backend services).  If you need
    additional business logic, consider using the CatalogClientHelper instead.
    """

    # !!!IMPORTANT!!!
    # Please use reformat_full_table_name() on feature table name field for all proto entities.

    def __init__(self, get_host_creds, feature_store_uri: Optional[str] = None):
        """
        Catalog client for the Feature Store client. Takes in an optional parameter to identify the remote workspace
        for multi-workspace Feature Store.
        :param feature_store_uri: An URI of the form ``databricks://<scope>.<prefix>`` that identifies the credentials
          of the intended Feature Store workspace. Throws an error if specified but credentials were not found.
        """
        self._get_host_creds = lambda: get_host_creds(feature_store_uri)
        self._local_host, self._local_workspace_id = self._get_local_workspace_info()
        (
            self._feature_store_workspace_host,
            self._feature_store_workspace_id,
        ) = self._get_feature_store_workspace_info(feature_store_uri)

    @property
    def feature_store_workspace_id(self) -> int:
        return self._feature_store_workspace_id

    def _get_local_workspace_info(self) -> (str, int):
        local_host, workspace_id = databricks_utils.get_workspace_info_from_dbutils()
        return local_host, self._parse_workspace_id(workspace_id)

    def _get_feature_store_workspace_info(
        self, feature_store_uri: Optional[str] = None
    ) -> (str, int):
        if feature_store_uri:
            # Retrieve the remote hostname and workspace ID stored in the secret scope by the user.
            (
                remote_hostname,
                remote_workspace_id,
            ) = databricks_utils.get_workspace_info_from_databricks_secrets(
                feature_store_uri
            )
            if not remote_workspace_id:
                raise ValueError(
                    f"'FeatureStoreClient' was initialized with 'feature_store_uri' argument "
                    f"for multi-workspace usage, but the remote Feature Store workspace ID was not "
                    f"found at URI {feature_store_uri}."
                )

            if not remote_hostname:
                raise ValueError(
                    f"'FeatureStoreClient' was initialized with 'feature_store_uri' argument "
                    f"for multi-workspace usage, but the remote Feature Store hostname was not "
                    f"found at URI {feature_store_uri}."
                )

            return remote_hostname, self._parse_workspace_id(remote_workspace_id)
        else:
            return self._local_host, self._local_workspace_id

    @staticmethod
    def _parse_workspace_id(workspace_id) -> int:
        if workspace_id:
            try:
                workspace_id = int(workspace_id)
            except (ValueError, TypeError):
                raise ValueError("Internal Error: Workspace ID was not found.")
        return workspace_id

    def _call_endpoint(self, api, json_body, req_context: RequestContext):
        endpoint, method = _METHOD_TO_INFO[api]
        response_proto = api.Response()
        return call_endpoint(
            self._get_host_creds(),
            endpoint,
            method,
            proto_to_json(json_body),
            response_proto,
            req_context,
        )

    def _get_feature_table(self, feature_table: str, req_context: RequestContext):
        req_body = GetFeatureTable(name=feature_table)
        return self._call_endpoint(GetFeatureTable, req_body, req_context)

    # CRUD API to call Feature Catalog
    def create_feature_table(
        self,
        feature_table: str,
        partition_key_spec,
        primary_key_spec,
        timestamp_key_spec,
        description: str,
        is_imported: str,
        req_context: RequestContext,
    ):
        req_body = CreateFeatureTable(
            name=reformat_full_table_name(feature_table),
            primary_keys=([key_spec.to_proto() for key_spec in primary_key_spec]),
            partition_keys=([key_spec.to_proto() for key_spec in partition_key_spec]),
            timestamp_keys=([key_spec.to_proto() for key_spec in timestamp_key_spec]),
            description=description,
            is_imported=is_imported,
        )
        response_proto = self._call_endpoint(CreateFeatureTable, req_body, req_context)
        return FeatureTable.from_proto(response_proto.feature_table)

    def get_feature_table(self, feature_table: str, req_context: RequestContext):
        response_proto = self._get_feature_table(
            feature_table=reformat_full_table_name(feature_table),
            req_context=req_context,
        )
        return FeatureTable.from_proto(response_proto.feature_table)

    def feature_table_exists(self, name: str, req_context: RequestContext):
        """
        Checks whether the feature table exists.

        This CatalogClient method is built on top of the feature-tables/get endpoint. There is no
        dedicated endpoint for feature_table_exists.
        """
        try:
            self.get_feature_table(name, req_context)
        except Exception as e:
            if get_error_code(e) == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                return False
            raise e
        return True

    def can_write_to_catalog(
        self, feature_table_name: str, req_context: RequestContext
    ):
        """
        Checks whether the user has write permission to feature catalog.

        This CatalogClient method is built on top of the feature-tables/get endpoint. There is no
        dedicated endpoint for can_write_to_catalog.
        """
        response_proto = self._get_feature_table(
            feature_table=reformat_full_table_name(feature_table_name),
            req_context=req_context,
        )
        return _PermissionLevel.can_write_to_catalog(
            response_proto.feature_table.permission_level
        )

    def publish_feature_table(
        self,
        feature_table: str,
        online_store_metadata: OnlineStoreMetadata,
        features: List[str],
        req_context: RequestContext,
    ):
        req_body = PublishFeatureTable(
            feature_table=reformat_full_table_name(feature_table),
            online_table=online_store_metadata.online_table,
            cloud=online_store_metadata.cloud,
            store_type=online_store_metadata.store_type,
            read_secret_prefix=online_store_metadata.read_secret_prefix,
            write_secret_prefix=online_store_metadata.write_secret_prefix,
            features=features,
        )
        if online_store_metadata.store_type == StoreType.MYSQL:
            req_body.mysql_metadata.CopyFrom(
                online_store_metadata.additional_metadata.to_proto()
            )
        elif online_store_metadata.store_type == StoreType.SQL_SERVER:
            req_body.sql_server_metadata.CopyFrom(
                online_store_metadata.additional_metadata.to_proto()
            )
        elif online_store_metadata.store_type == StoreType.DYNAMODB:
            req_body.dynamodb_metadata.CopyFrom(
                online_store_metadata.additional_metadata.to_proto()
            )
        elif online_store_metadata.store_type == StoreType.COSMOSDB:
            req_body.cosmosdb_metadata.CopyFrom(
                online_store_metadata.additional_metadata.to_proto()
            )
        else:
            raise TypeError(
                f"Unsupported online store metadata type {online_store_metadata.additional_metadata}"
            )

        response_proto = self._call_endpoint(PublishFeatureTable, req_body, req_context)
        return OnlineStoreDetailed.from_proto(response_proto.online_store)

    def create_features(
        self, feature_table: str, feature_specs, req_context: RequestContext
    ):
        req_body = CreateFeatures(
            feature_table=reformat_full_table_name(feature_table),
            features=[key_spec.to_proto() for key_spec in feature_specs],
        )
        self._call_endpoint(CreateFeatures, req_body, req_context)

    def get_features(self, feature_table: str, req_context: RequestContext):
        all_features = []
        page_token = None
        while True:
            # Use default max_results
            req_body = GetFeatures(
                feature_table=reformat_full_table_name(feature_table),
                page_token=page_token,
            )
            response_proto = self._call_endpoint(GetFeatures, req_body, req_context)
            all_features += [
                Feature.from_proto(feature) for feature in response_proto.features
            ]
            page_token = response_proto.next_page_token
            if not page_token:
                break
        return all_features

    def update_feature(
        self,
        feature_table: str,
        feature: str,
        data_type: str,
        req_context: RequestContext,
    ):
        req_body = UpdateFeature(
            feature_table=reformat_full_table_name(feature_table),
            name=feature,
            data_type=data_type.upper(),
        )
        response_body = self._call_endpoint(UpdateFeature, req_body, req_context)
        return Feature.from_proto(response_body.feature)

    def delete_feature_table(
        self, feature_table: str, req_context: RequestContext, dry_run=False
    ):
        req_body = DeleteFeatureTable(
            name=reformat_full_table_name(feature_table), dry_run=dry_run
        )
        self._call_endpoint(DeleteFeatureTable, req_body, req_context)

    def add_data_sources(
        self,
        feature_table: str,
        tables: List[str],
        paths: List[str],
        custom_sources: List[str],
        req_context: RequestContext,
    ):
        req_body = AddDataSources(
            feature_table=reformat_full_table_name(feature_table),
            tables=tables,
            paths=paths,
            custom_sources=custom_sources,
        )
        self._call_endpoint(AddDataSources, req_body, req_context)

    def delete_data_sources(
        self,
        feature_table: str,
        source_names: List[str],
        req_context: RequestContext,
    ):
        req_body = DeleteDataSources(
            feature_table=reformat_full_table_name(feature_table), sources=source_names
        )
        self._call_endpoint(DeleteDataSources, req_body, req_context)

    def add_notebook_producer(
        self,
        feature_table: str,
        notebook_id: int,
        revision_id: int,
        producer_action: ProducerAction,
        req_context: RequestContext,
    ):
        notebook = Notebook(
            notebook_id=notebook_id,
            revision_id=revision_id,
            notebook_workspace_id=self._local_workspace_id,
            notebook_workspace_url=self._local_host,
        )
        req_body = AddProducer(
            feature_table=reformat_full_table_name(feature_table),
            notebook=notebook,
            producer_action=producer_action,
        )
        self._call_endpoint(AddProducer, req_body, req_context)

    def add_job_producer(
        self,
        feature_table: str,
        job_id: int,
        run_id: int,
        producer_action: ProducerAction,
        req_context: RequestContext,
    ):
        job = Job(
            job_id=job_id,
            run_id=run_id,
            job_workspace_id=self._local_workspace_id,
            job_workspace_url=self._local_host,
        )
        req_body = AddProducer(
            feature_table=reformat_full_table_name(feature_table),
            job_run=job,
            producer_action=producer_action,
        )
        self._call_endpoint(AddProducer, req_body, req_context)

    def add_notebook_consumer(
        self,
        feature_table_map: Dict[str, List[str]],
        notebook_id: int,
        revision_id: int,
        req_context: RequestContext,
    ):
        features = [
            ConsumedFeatures(
                table=reformat_full_table_name(feature_table),
                names=features,
            )
            for feature_table, features in feature_table_map.items()
        ]
        notebook = Notebook(
            notebook_id=notebook_id,
            revision_id=revision_id,
            notebook_workspace_id=self._local_workspace_id,
            notebook_workspace_url=self._local_host,
        )
        req_body = AddConsumer(
            features=features,
            notebook=notebook,
        )
        self._call_endpoint(AddConsumer, req_body, req_context)

    def add_job_consumer(
        self,
        feature_table_map: Dict[str, List[str]],
        job_id: int,
        run_id: int,
        req_context: RequestContext,
    ):
        features = [
            ConsumedFeatures(
                table=reformat_full_table_name(feature_table),
                names=features,
            )
            for feature_table, features in feature_table_map.items()
        ]
        job = Job(
            job_id=job_id,
            run_id=run_id,
            job_workspace_id=self._local_workspace_id,
            job_workspace_url=self._local_host,
        )
        req_body = AddConsumer(
            features=features,
            job_run=job,
        )
        self._call_endpoint(AddConsumer, req_body, req_context)

    def get_consumers(self, feature_table: str, req_context: RequestContext):
        req_body = GetConsumers(feature_table=reformat_full_table_name(feature_table))
        response_proto = self._call_endpoint(GetConsumers, req_body, req_context)
        return [Consumer.from_proto(consumer) for consumer in response_proto.consumers]

    def get_model_serving_metadata(
        self,
        feature_table_to_features: Dict[str, List[str]],
        req_context: RequestContext,
    ):
        req_body = GetModelServingMetadata(
            feature_table_features=[
                FeatureTableFeatures(
                    feature_table_name=reformat_full_table_name(ft_name),
                    features=features,
                )
                for (ft_name, features) in feature_table_to_features.items()
            ]
        )
        response_proto = self._call_endpoint(
            GetModelServingMetadata, req_body, req_context
        )
        return [
            OnlineFeatureTable.from_proto(online_ft)
            for online_ft in response_proto.online_feature_tables
        ]

    def set_feature_table_tags(
        self, feature_table_id: str, tags: Dict[str, str], req_context: RequestContext
    ) -> None:
        proto_tags = [ProtoTag(key=key, value=value) for key, value in tags.items()]
        req_body = SetTags(feature_table_id=feature_table_id, tags=proto_tags)
        self._call_endpoint(SetTags, req_body, req_context)

    def delete_feature_table_tags(
        self, feature_table_id: str, tags: List[str], req_context: RequestContext
    ) -> None:
        req_body = DeleteTags(feature_table_id=feature_table_id, keys=tags)
        self._call_endpoint(DeleteTags, req_body, req_context)

    def get_feature_table_tags(
        self, feature_table_id: str, req_context: RequestContext
    ) -> List[Tag]:
        req_body = GetTags(feature_table_id=feature_table_id)
        response_proto = self._call_endpoint(GetTags, req_body, req_context)
        return [Tag.from_proto(tag_proto) for tag_proto in response_proto.tags]

    def get_online_store(
        self,
        feature_table: str,
        online_store_metadata: OnlineStoreMetadata,
        req_context: RequestContext,
    ):
        req_body = GetOnlineStore(
            feature_table=reformat_full_table_name(feature_table),
            online_table=online_store_metadata.online_table,
            cloud=online_store_metadata.cloud,
            store_type=online_store_metadata.store_type,
        )
        if online_store_metadata.store_type == StoreType.DYNAMODB:
            req_body.table_arn = online_store_metadata.additional_metadata.table_arn
        elif online_store_metadata.store_type == StoreType.COSMOSDB:
            req_body.container_uri = (
                online_store_metadata.additional_metadata.container_uri
            )
        response_proto = self._call_endpoint(GetOnlineStore, req_body, req_context)
        return OnlineStoreDetailed.from_proto(response_proto.online_store)
