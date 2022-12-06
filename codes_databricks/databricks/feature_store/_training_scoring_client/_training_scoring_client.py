import logging
import os
from typing import List, Union, Dict, Any, Optional, Set
from types import ModuleType
from collections import Counter, defaultdict

from databricks.feature_store.constants import (
    PREDICTION_COLUMN_NAME,
    MODEL_DATA_PATH_ROOT,
    _WARN,
)
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from databricks.feature_store.entities.feature import Feature
from databricks.feature_store.entities.feature_column_info import FeatureColumnInfo
from databricks.feature_store.entities.feature_table_info import FeatureTableInfo
from databricks.feature_store.entities.feature_spec import FeatureSpec
from databricks.feature_store.entities.feature_table import FeatureTable
from databricks.feature_store.entities.source_data_column_info import (
    SourceDataColumnInfo,
)
from databricks.feature_store.hive_client import HiveClient
from databricks.feature_store.catalog_client import CatalogClient
from databricks.feature_store import mlflow_model_constants
from databricks.feature_store.training_set import TrainingSet
from databricks.feature_store.utils import utils
from databricks.feature_store.utils.feature_lookup_utils import join_feature_data
from databricks.feature_store.utils import uc_utils
from databricks.feature_store.utils import validation_utils
from databricks.feature_store.utils import request_context
from databricks.feature_store.utils.request_context import (
    RequestContext,
)
from databricks.feature_store.version import VERSION
from databricks.feature_store.utils import schema_utils
from databricks.feature_store._hive_client_helper import HiveClientHelper
from databricks.feature_store._catalog_client_helper import CatalogClientHelper

from pyspark.sql import DataFrame
from pyspark.sql.functions import struct

from mlflow.models import Model
from mlflow.utils.file_utils import TempDir, read_yaml
import mlflow

_logger = logging.getLogger(__name__)


class TrainingScoringClient:
    def __init__(
        self,
        catalog_client: CatalogClient,
        catalog_client_helper: CatalogClientHelper,
        hive_client: HiveClient,
        model_registry_uri: str,
    ):
        self._catalog_client = catalog_client
        self._catalog_client_helper = catalog_client_helper
        self._hive_client = hive_client
        self._hive_client_helper = HiveClientHelper(self._hive_client)
        self._model_registry_uri = model_registry_uri

    def create_training_set(
        self,
        df: DataFrame,
        feature_lookups: List[FeatureLookup],
        label: Union[str, List[str], None],
        exclude_columns: List[str] = [],
    ) -> TrainingSet:

        req_context = RequestContext(request_context.CREATE_TRAINING_SET)

        validation_utils.check_dataframe_type(df)

        # Verify DataFrame does not have duplicate columns
        df_column_name_counts = Counter(df.columns)
        for column_name in df.columns:
            if df_column_name_counts[column_name] > 1:
                raise ValueError(
                    f"Found duplicate DataFrame column name '{column_name}'"
                )

        # Initialize label_names with empty list if label is not provided
        label_names = utils.as_list(label, [])

        # Verify that label is in DataFrame and not in exclude_columns
        for label_name in label_names:
            if label_name not in df.columns:
                raise ValueError(
                    f"Label column '{label_name}' was not found in DataFrame"
                )
            if label_name in exclude_columns:
                raise ValueError(f"Label column '{label_name}' cannot be excluded")

        table_names = set([fl.table_name for fl in feature_lookups])

        feature_table_features_map = self._get_feature_names_for_tables(
            req_context, table_names=table_names
        )
        feature_table_metadata_map = self._get_feature_table_metadata_for_tables(
            req_context, table_names=table_names
        )
        feature_table_data_map = self._load_feature_data_for_tables(
            table_names=table_names
        )

        # Collect FeatureColumnInfos
        feature_column_infos = self._explode_feature_lookups(
            feature_lookups, feature_table_features_map, feature_table_metadata_map
        )

        feature_output_names = [
            feature_info.output_name for feature_info in feature_column_infos
        ]

        # Verify features have unique output names
        feature_output_names_counts = Counter(feature_output_names)
        for output_name in feature_output_names_counts:
            if feature_output_names_counts[output_name] > 1:
                raise ValueError(f"Found duplicate feature output name '{output_name}'")

        # Verify labels do not collide with feature output names
        for label_name in label_names:
            if label_name in feature_output_names:
                raise ValueError(
                    f"Feature cannot have same output name as label '{label_name}'"
                )

        # Verify that DataFrame does not override features requested in FeatureLookup list
        overridden_features = [
            column_name
            for column_name in df.columns
            if column_name in feature_output_names
        ]
        if len(overridden_features) > 0:
            overriden_features_names = ", ".join(overridden_features)
            raise ValueError(
                f"DataFrame contains column names that match feature output names specified"
                f" in FeatureLookups '{overriden_features_names}'. Either remove these columns"
                f" from the DataFrame or FeatureLookups."
            )

        # Collect SourceDataColumnInfo
        source_columns_infos = [
            SourceDataColumnInfo(column_name)
            for column_name in df.columns
            if column_name not in feature_output_names
            and column_name not in label_names
        ]

        column_infos = [
            ci
            for ci in (source_columns_infos + feature_column_infos)
            if ci.output_name not in exclude_columns
        ]

        # Keep table_infos sorted by table name, so they appear sorted in feature_spec.yaml
        table_infos = sorted(
            [
                FeatureTableInfo(
                    table_name=feature_table_name,
                    table_id=feature_table_metadata.table_id,
                )
                for feature_table_name, feature_table_metadata in feature_table_metadata_map.items()
            ],
            key=lambda table_info: table_info.table_name,
        )

        workspace_id = self._catalog_client.feature_store_workspace_id

        # Build FeatureSpec
        feature_spec = FeatureSpec(column_infos, table_infos, workspace_id, VERSION)

        # Validate FeatureSpec
        self._validate_data_for_feature_spec(
            feature_spec.feature_column_infos,
            feature_table_features_map,
            feature_table_data_map,
        )

        # TODO(divyagupta-db): Move validation from _validate_join_feature_data in feature_lookup_utils.py
        #  to a helper function called here and in score_batch.

        # Add consumer of each feature as final step
        consumer_feature_table_map = defaultdict(list)
        for feature in feature_column_infos:
            consumer_feature_table_map[feature.table_name].append(feature.feature_name)
        self._catalog_client_helper.add_consumers(
            consumer_feature_table_map, req_context
        )

        return TrainingSet(
            feature_spec,
            df,
            label_names,
            feature_table_metadata_map,
            feature_table_data_map,
        )

    def score_batch(
        self, model_uri: str, df: DataFrame, result_type: str = "double"
    ) -> DataFrame:
        req_context = RequestContext(request_context.SCORE_BATCH)

        validation_utils.check_dataframe_type(df)

        if PREDICTION_COLUMN_NAME in df.columns:
            raise ValueError(
                "FeatureStoreClient.score_batch returns a DataFrame with a new column "
                f'"{PREDICTION_COLUMN_NAME}". df already has a column with name '
                f'"{PREDICTION_COLUMN_NAME}".'
            )

        # The call to mlflow.set_registry_uri sets the model registry globally, so it needs
        # to be set appropriately each time. If no model_registry_uri is specified, we pass
        # in an empty string to reset the model registry to the local one.
        mlflow.set_registry_uri(self._model_registry_uri or "")

        artifact_path = os.path.join(mlflow.pyfunc.DATA, MODEL_DATA_PATH_ROOT)

        with TempDir() as tmp_location:
            local_path = utils.download_model_artifacts(model_uri, tmp_location.path())
            model_data_path = os.path.join(local_path, artifact_path)
            # Augment local workspace metastore tables from 2L to 3L,
            # this will prevent us from erroneously reading data from other catalogs
            feature_spec = uc_utils.get_feature_spec_with_full_table_names(
                FeatureSpec.load(model_data_path)
            )
            raw_model_path = os.path.join(
                model_data_path, mlflow_model_constants.RAW_MODEL_FOLDER
            )
            predict_udf = self._hive_client.get_predict_udf(
                raw_model_path, result_type=result_type
            )
            # TODO (ML-17260) Consider reading the timestamp from the backend instead of feature store artifacts
            ml_model = Model.load(
                os.path.join(local_path, mlflow_model_constants.ML_MODEL)
            )
            model_creation_timestamp_ms = (
                utils.utc_timestamp_ms_from_iso_datetime_string(
                    ml_model.utc_time_created
                )
            )

        # Validate that columns needed for joining feature tables exist and are not duplicates.
        required_cols = []
        for column_info in feature_spec.column_infos:
            col_info_req_cols = (
                [column_info.name]
                if isinstance(column_info, SourceDataColumnInfo)
                else column_info.lookup_key
            )
            for col in col_info_req_cols:
                if col not in required_cols:
                    required_cols.append(col)

        df_column_name_counts = Counter(df.columns)
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"DataFrame is missing column '{col}'. The following columns are required: "
                    f"{required_cols}."
                )
            if df_column_name_counts[col] > 1:
                raise ValueError(
                    f"Columns required for scoring or looking up features cannot be "
                    f"duplicates. Found duplicate column '{col}'."
                )

        table_names = set([fci.table_name for fci in feature_spec.feature_column_infos])

        feature_table_features_map = self._get_feature_names_for_tables(
            req_context, table_names=table_names
        )
        feature_table_metadata_map = self._get_feature_table_metadata_for_tables(
            req_context, table_names=table_names
        )
        feature_table_data_map = self._load_feature_data_for_tables(
            table_names=table_names
        )

        self._validate_data_for_feature_spec(
            feature_spec.feature_column_infos,
            feature_table_features_map,
            feature_table_data_map,
        )

        # Check if the fetched feature tables match the feature tables logged in training

        # 1. Compare feature table ids
        # Check for feature_spec logged with client versions that supports table_infos
        if len(feature_spec.table_infos) > 0:
            # When feature_spec.yaml is parsed, FeatureSpec.load will assert
            # that the listed table names in input_tables match table names in input_columns.
            # The following code assumes this as invariant and only checks for the table IDs.
            mismatched_tables = []
            for table_info in feature_spec.table_infos:
                feature_table = feature_table_metadata_map[table_info.table_name]
                if feature_table and table_info.table_id != feature_table.table_id:
                    mismatched_tables.append(table_info.table_name)
            if len(mismatched_tables) > 0:
                plural = len(mismatched_tables) > 1
                _logger.warning(
                    f"Feature table{'s' if plural else ''} {', '.join(mismatched_tables)} "
                    f"{'were' if plural else 'was'} deleted and recreated after "
                    f"the model was trained. Model performance may be affected if the features "
                    f"used in scoring have drifted from the features used in training."
                )

        # 2. Compare model creation timestamp with feature table creation timestamps
        feature_tables_created_after_model = []
        for name, metadata in feature_table_metadata_map.items():
            if model_creation_timestamp_ms < metadata.creation_timestamp:
                feature_tables_created_after_model.append(name)
        if len(feature_tables_created_after_model) > 0:
            plural = len(feature_tables_created_after_model) > 1
            message = (
                f"Feature table{'s' if plural else ''} {', '.join(feature_tables_created_after_model)} "
                f"{'were' if plural else 'was'} created after the model was logged. "
                f"Model performance may be affected if the features used in scoring have drifted "
                f"from the features used in training."
            )
            _logger.warning(message)

        df_with_features = join_feature_data(
            feature_spec,
            df,
            extra_columns=df.columns,
            feature_table_metadata_map=feature_table_metadata_map,
            feature_table_data_map=feature_table_data_map,
        )

        udf_input_columns = [c.output_name for c in feature_spec.column_infos]
        udf_inputs = struct(*udf_input_columns)

        df_with_predictions = df_with_features.withColumn(
            PREDICTION_COLUMN_NAME, predict_udf(udf_inputs)
        )

        # Reorder dataframe columns
        df_columns = [col for col in df.columns]
        extra_feature_spec_columns = [
            col for col in udf_input_columns if col not in df_columns
        ]
        prediction_column = [PREDICTION_COLUMN_NAME]

        return_value = df_with_predictions.select(
            df_columns + extra_feature_spec_columns + prediction_column
        )

        # Add consumer of each feature as final step
        consumer_feature_table_map = defaultdict(list)
        for feature in feature_spec.feature_column_infos:
            consumer_feature_table_map[feature.table_name].append(feature.feature_name)
        self._catalog_client_helper.add_consumers(
            consumer_feature_table_map, req_context
        )

        return return_value

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
        # Validate only one of the training_set, feature_spec_path arguments is provided.
        # Retrieve the FeatureSpec, then remove training_set, feature_spec_path from local scope.
        feature_spec_path = kwargs.pop("feature_spec_path", None)
        if (training_set is None) == (feature_spec_path is None):
            raise ValueError(
                "Either 'training_set' or 'feature_spec_path' must be provided, but not both."
            )
        if training_set:
            # Reformat tables in local metastore to 2L before serialization, this will make sure
            # the format of the feature spec with local metastore tables is always consistent.
            feature_spec = uc_utils.get_feature_spec_with_reformat_full_table_names(
                training_set.feature_spec
            )
        else:
            # FeatureSpec.load expects the root directory of feature_spec.yaml
            root_dir, file_name = os.path.split(feature_spec_path)
            if file_name != FeatureSpec.FEATURE_ARTIFACT_FILE:
                raise ValueError(
                    f"'feature_spec_path' must be a path to {FeatureSpec.FEATURE_ARTIFACT_FILE}."
                )
            feature_spec = FeatureSpec.load(root_dir)
        del training_set, feature_spec_path

        with TempDir() as tmp_location:
            data_path = os.path.join(tmp_location.path(), "feature_store")
            raw_mlflow_model = Model()
            raw_model_path = os.path.join(
                data_path, mlflow_model_constants.RAW_MODEL_FOLDER
            )
            if flavor.FLAVOR_NAME != mlflow.pyfunc.FLAVOR_NAME:
                flavor.save_model(
                    model, raw_model_path, mlflow_model=raw_mlflow_model, **kwargs
                )
            else:
                flavor.save_model(
                    raw_model_path,
                    mlflow_model=raw_mlflow_model,
                    python_model=model,
                    **kwargs,
                )
            if not "python_function" in raw_mlflow_model.flavors:
                raise ValueError(
                    f"FeatureStoreClient.log_model does not support '{flavor.__name__}' "
                    f"since it does not have a python_function model flavor."
                )

            # Re-use the conda environment from the raw model for the packaged model. Later, we may
            # add an additional requirement for the Feature Store library. At the moment, however,
            # the databricks-feature-store package is not available via conda or pip.
            conda_file = raw_mlflow_model.flavors["python_function"][mlflow.pyfunc.ENV]
            conda_env = read_yaml(raw_model_path, conda_file)

            # Get the pip package string for the databricks-feature-lookup client
            databricks_feature_lookup_pip_package = utils.pip_depependency_pinned_major_version(
                pip_package_name=mlflow_model_constants.FEATURE_LOOKUP_CLIENT_PIP_PACKAGE,
                major_version=mlflow_model_constants.FEATURE_LOOKUP_CLIENT_MAJOR_VERSION,
            )

            # Add pip dependencies required for online feature lookups
            utils.add_mlflow_pip_depependency(
                conda_env, databricks_feature_lookup_pip_package
            )

            feature_spec.save(data_path)

            # Log the packaged model. If no run is active, this call will create an active run.
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                loader_module=mlflow_model_constants.MLFLOW_MODEL_NAME,
                data_path=data_path,
                code_path=None,
                conda_env=conda_env,
            )
        if registered_model_name is not None:
            # The call to mlflow.pyfunc.log_model will create an active run, so it is safe to
            # obtain the run_id for the active run.
            run_id = mlflow.tracking.fluent.active_run().info.run_id

            # The call to mlflow.set_registry_uri sets the model registry globally, so it needs
            # to be set appropriately each time. If no model_registry_uri is specified, we pass
            # in an empty string to reset the model registry to the local one.
            mlflow.set_registry_uri(self._model_registry_uri or "")
            mlflow.register_model(
                "runs:/%s/%s" % (run_id, artifact_path),
                registered_model_name,
                await_registration_for=await_registration_for,
            )

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

    def _validate_data_for_feature_spec(
        self,
        feature_column_infos: List[FeatureColumnInfo],
        features_by_table: Dict[str, List[Feature]],
        feature_table_data_map: Dict[str, DataFrame],
    ):
        """
        Validate that

        1) Feature tables exist in Delta
        2) Features exist in Delta and Feature Catalog
        3) Feature types match in Delta and Feature Catalog
        """

        table_names = set([fi.table_name for fi in feature_column_infos])

        # Validate that all feature tables exists to fail-fast before API requests
        for table_name in table_names:
            self._hive_client_helper.check_feature_table_exists(table_name)

        table_data = {}
        for table_name in table_names:
            feature_names_in_spec = [
                info.feature_name
                for info in feature_column_infos
                if info.table_name == table_name
            ]

            catalog_features = features_by_table[table_name]

            feature_table_data = feature_table_data_map[table_name]

            catalog_schema = {
                feature.name: feature.data_type for feature in catalog_features
            }
            delta_schema = {
                feature.name: feature.dataType.typeName().upper()
                for feature in feature_table_data.schema
            }

            for feature_name in feature_names_in_spec:
                if feature_name not in catalog_schema:
                    raise ValueError(
                        f"Unable to find feature '{feature_name}' from feature table '{table_name}' in Feature Catalog."
                    )
                if feature_name not in delta_schema:
                    raise ValueError(
                        f"Unable to find feature '{feature_name}' from feature table '{table_name}' in Delta."
                    )
                if catalog_schema[feature_name] != delta_schema[feature_name]:
                    raise ValueError(
                        f"Expected type of feature '{feature_name}' from feature table '{table_name}' "
                        f"to be equivalent in Feature Catalog and Delta. "
                        f"Feature has type '{catalog_schema[feature_name]}' in Feature Catalog and "
                        f"'{delta_schema[feature_name]}' in Delta."
                    )

            # Warn if mismatch in other features in feature table
            if not schema_utils.catalog_matches_delta_schema(
                catalog_features, feature_table_data.schema
            ):
                schema_utils.log_catalog_schema_not_match_delta_schema(
                    catalog_features,
                    feature_table_data.schema,
                    level=_WARN,
                )
