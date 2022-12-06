from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import List, Dict
import itertools
import os

from mlflow import sagemaker as mlflow_sagemaker
from mlflow.utils import databricks_utils
from mlflow.pyfunc import DATA

from databricks.feature_store.catalog_client import CatalogClient
from databricks.feature_store.constants import MODEL_DATA_PATH_ROOT
from databricks.feature_store.entities.feature_spec import FeatureSpec
from databricks.feature_store.entities.feature_tables_for_serving import (
    FeatureTablesForSageMakerServing,
)
from databricks.feature_store.entities.online_feature_table import (
    OnlineFeatureTable,
    OnlineFeatureTableForSageMakerServing,
)
from databricks.feature_store.entities.online_store_for_serving import (
    OnlineStoreForSageMakerServing,
)

from databricks.feature_store.entities.store_type import StoreType
from databricks.feature_store.utils import request_context
from databricks.feature_store.utils.request_context import RequestContext
from databricks.feature_store.utils.utils import download_model_artifacts
from databricks.feature_store.utils import uc_utils


def _convert_olft_to_sagemaker_olft(
    olft: OnlineFeatureTable,
) -> OnlineFeatureTableForSageMakerServing:
    """
    Transforms an OnlineFeatureTable to an OnlineFeatureTableForSageMakerServing

    :param olft: The source OnlineFeatureTable. Must have store_type DYNAMODB.
    """
    online_store = olft.online_store
    if not online_store.store_type == StoreType.DYNAMODB:
        raise Exception(
            "Internal Error: Attempted to convert online feature table with store type "
            f" {online_store.store_type} to OnlineFeatureTableForSageMakerServing."
        )
    online_store_for_sagemaker_serving = OnlineStoreForSageMakerServing(
        creation_timestamp_ms=online_store.creation_timestamp_ms,
        # Only DynamoDB online stores are expected at this point, so extra_configs
        # is a DynamoDbConf.
        extra_configs=online_store.extra_configs,
        query_mode=online_store.query_mode,
    )
    return OnlineFeatureTableForSageMakerServing(
        feature_table_name=olft.feature_table_name,
        online_feature_table_name=olft.online_feature_table_name,
        online_store=online_store_for_sagemaker_serving,
        primary_keys=olft.primary_keys,
        feature_table_id=olft.feature_table_id,
        features=olft.features,
        timestamp_keys=olft.timestamp_keys,
    )


def _generate_sagemaker_model_serving_metadata(
    feature_table_to_features: Dict[str, List[str]],
    online_feature_tables: List[OnlineFeatureTable],
) -> FeatureTablesForSageMakerServing:
    """
    Generates online feature table metadata required by SageMaker serving.

    :param feature_table_to_features: Dictionary of required features per feature table. eg
      {"prod.user_features": ["age", "sex", "bmi", ...]
    :param online_feature_tables: Online feature tables available, returned by
      GetModelServingMetdata
    :return: FeatureTablesForSageMakerServing object
    """
    # Group candidate online feature tables by feature table name
    feature_table_to_online_feature_tables = {
        k: list(v)
        for k, v in itertools.groupby(
            online_feature_tables, lambda olft: olft.feature_table_name
        )
    }
    # For each required feature table, select up to one online feature table
    selected_online_feature_table_or_none = {}
    for ft_name in feature_table_to_features.keys():
        all_olfts = feature_table_to_online_feature_tables.get(ft_name, [])
        supported_olfts = [
            olft
            for olft in all_olfts
            if olft.online_store.store_type == StoreType.DYNAMODB
        ]
        # If there are multiple options, choose the oldest
        ts_sorted_olfts = sorted(
            supported_olfts, key=lambda olft: olft.online_store.creation_timestamp_ms
        )
        selected_online_feature_table_or_none[ft_name] = (
            ts_sorted_olfts[0] if ts_sorted_olfts else None
        )

    # Throw if there is no online feature table
    missing_online_feature_tables = [
        ft
        for (ft, olft) in selected_online_feature_table_or_none.items()
        if olft is None
    ]
    if missing_online_feature_tables:
        missing_table_strs = [
            f"\n\tFeature Table: {ft_name}\n\t\tRequired Features:{feature_table_to_features[ft_name]}"
            for ft_name in missing_online_feature_tables
        ]
        missing_tables_str = "\n\n".join(missing_table_strs)
        feature_table_str = (
            "feature_table"
            if len(missing_online_feature_tables) <= 1
            else "feature_tables"
        )
        raise ValueError(
            f"No valid online stores were found for {feature_table_str} {missing_online_feature_tables}. "
            "In order to deploy a model to SageMaker, feature tables must be published to a DynamoDB online store, "
            "including all required features. The following feature tables were not published to DynamoDB or are missing"
            " features in the online store:"
            f"{missing_tables_str}"
        )

    online_feature_tables_for_sagemaker_serving = [
        _convert_olft_to_sagemaker_olft(olft)
        for olft in selected_online_feature_table_or_none.values()
    ]

    return FeatureTablesForSageMakerServing(
        online_feature_tables=online_feature_tables_for_sagemaker_serving
    )


def _feature_spec_to_feature_table_to_features(feature_spec: FeatureSpec):
    feature_table_to_features = defaultdict(list)
    for fci in feature_spec.feature_column_infos:
        feature_table_to_features[fci.table_name].append(fci.feature_name)
    return feature_table_to_features


def _add_model_serving_artifact(model_data_path: str):
    """
    Retrieve and save model serving metadata from the Feature Catalog.

    :param model_data_path: Path to the model's data directory. eg
      "/tmp/the_model/data/feature_store"
    """
    feature_spec = FeatureSpec.load(model_data_path)
    feature_spec_with_full_table_names = (
        uc_utils.get_feature_spec_with_full_table_names(feature_spec)
    )
    feature_table_to_features = _feature_spec_to_feature_table_to_features(feature_spec)

    if feature_table_to_features:
        catalog_client = CatalogClient(databricks_utils.get_databricks_host_creds)
        # use full table name when interacting with catalog client
        online_feature_tables = catalog_client.get_model_serving_metadata(
            _feature_spec_to_feature_table_to_features(
                feature_spec_with_full_table_names
            ),
            req_context=RequestContext(request_context.GET_MODEL_SERVING_METADATA),
        )
    else:
        online_feature_tables = []

    sagemaker_model_serving_metadata = _generate_sagemaker_model_serving_metadata(
        feature_table_to_features, online_feature_tables
    )
    s = sagemaker_model_serving_metadata.to_proto().SerializeToString()

    with open(
        os.path.join(model_data_path, FeatureTablesForSageMakerServing.DATA_FILE), "wb"
    ) as f:
        f.write(s)


def deploy(app_name: str, model_uri: str, execution_role_arn: str = None, **kwargs):
    """
    Deploy an MLflow model on AWS SageMaker. The IAM role that your Databricks cluster is running
    under must have sufficient permissions to deploy a SageMaker model.

    The model must have been logged with :meth:`FeatureStoreClient.log_model`.

    :param app_name: Name of the deployed application.
    :param model_uri: The location, in URI format, of the MLflow model logged using
          :meth:`FeatureStoreClient.log_model`. One of:

            * ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            * ``models:/<model_name>/<model_version>``

            * ``models:/<model_name>/<stage>``

    :param execution_role_arn: The name of an IAM role granting the SageMaker service permissions to
                               access the specified Docker image and S3 bucket containing MLflow
                               model artifacts. If unspecified, the currently-assumed role will be
                               used. This execution role is passed to the SageMaker service when
                               creating a SageMaker model from the specified MLflow model. It is
                               passed as the ``ExecutionRoleArn`` parameter of the `SageMaker
                               CreateModel API call <https://docs.aws.amazon.com/sagemaker/latest/
                               dg/API_CreateModel.html>`_. This role is *not* assumed for any other
                               call. For more information about SageMaker execution roles for model
                               creation, see
                               https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html.

    Additional keyword arguments will be passed through to :func:`mlflow.sagemaker.deploy`.
    """
    with TemporaryDirectory() as tmp_dir:
        download_model_artifacts(model_uri, tmp_dir)
        # Augment model model artifacts with an additional model serving artifact. This will be
        # used by SageMaker to identify and connect to online stores for feature lookup.
        _add_model_serving_artifact(
            model_data_path=os.path.join(tmp_dir, DATA, MODEL_DATA_PATH_ROOT)
        )
        mlflow_sagemaker.deploy(
            app_name=app_name,
            model_uri=tmp_dir,
            execution_role_arn=execution_role_arn,
            **kwargs,
        )
