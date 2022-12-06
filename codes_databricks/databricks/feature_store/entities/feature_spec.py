import os

from mlflow.utils.file_utils import TempDir

from databricks.feature_store.entities.feature_column_info import FeatureColumnInfo
from databricks.feature_store.entities.feature_table_info import FeatureTableInfo
from databricks.feature_store.entities.source_data_column_info import (
    SourceDataColumnInfo,
)
from databricks.feature_store.protos.feature_spec_pb2 import (
    FeatureSpec as ProtoFeatureSpec,
    ColumnInfo as ProtoColumnInfo,
)
from databricks.feature_store.utils import file_utils as fs_file_utils

from google.protobuf.json_format import MessageToDict, ParseDict

import mlflow
from mlflow.utils import file_utils as mlflow_file_utils

from typing import List, Union

from databricks.feature_store.entities.feature_spec_constants import (
    FEATURE_COLUMN_INFO,
    FEATURE_STORE,
    INPUT_COLUMNS,
    NAME,
    OUTPUT_NAME,
    SOURCE,
    SOURCE_DATA_COLUMN_INFO,
    TRAINING_DATA,
    SERIALIZATION_VERSION,
    INPUT_TABLES,
    TABLE_NAME,
    TABLE_ID,
)
from databricks.feature_store.utils.utils_common import is_artifact_uri

# Change log for serialization version
# Please update for each serialization version.
# 1. initial.
# 2. (2021/06/16): Record feature_store_client_version to help us
# making backward compatible changes in the future.
# 3. (2021/08/25): Record table_id to handle feature table lineage stability if tables are deleted.
# 4. (2021/09/25): Record timestamp_lookup_key to handle point-in-time lookups.


class FeatureSpec:

    FEATURE_ARTIFACT_FILE = "feature_spec.yaml"
    SERIALIZATION_VERSION_NUMBER = 4

    def __init__(
        self,
        column_infos: List[Union[FeatureColumnInfo, SourceDataColumnInfo]],
        table_infos: List[FeatureTableInfo],
        workspace_id: int,
        feature_store_client_version: str,
    ):
        if not column_infos:
            raise ValueError("column_infos must be non-empty.")

        for column_info in column_infos:
            if not isinstance(column_info, (FeatureColumnInfo, SourceDataColumnInfo)):
                raise ValueError(
                    f"Expected all elements of column_infos to be instances of SourceDataColumnInfo"
                    f" or FeatureColumnInfo. '{column_info}' is of the wrong type."
                )

        self._column_infos = column_infos
        # table_infos must be present
        if table_infos is None:
            raise ValueError("table_infos must be provided.")
        # The mapping of feature table name to feature table ids.
        self._table_infos = table_infos
        self._workspace_id = workspace_id
        # The Feature Store python client version which wrote this FeatureSpec. If empty, the
        # version is <=0.3.1.
        self._feature_store_client_version = feature_store_client_version

    def __eq__(self, other):
        if not isinstance(other, FeatureSpec):
            return False
        return self.__dict__ == other.__dict__

    @property
    def column_infos(self):
        return self._column_infos

    @property
    def table_infos(self):
        return self._table_infos

    @property
    def source_data_column_infos(self) -> List[SourceDataColumnInfo]:
        return [
            col_info
            for col_info in self._column_infos
            if isinstance(col_info, SourceDataColumnInfo)
        ]

    @property
    def feature_column_infos(self) -> List[FeatureColumnInfo]:
        return [
            col_info
            for col_info in self._column_infos
            if isinstance(col_info, FeatureColumnInfo)
        ]

    @property
    def workspace_id(self):
        return self._workspace_id

    @classmethod
    def from_proto(cls, feature_spec_proto):
        # Serialization version is not deserialized from the proto as there is currently only one
        # possible version.
        column_infos = []
        for proto_column_info in feature_spec_proto.input_columns:
            if proto_column_info.HasField(SOURCE_DATA_COLUMN_INFO):
                column_infos.append(
                    SourceDataColumnInfo.from_proto(
                        proto_column_info.source_data_column_info
                    )
                )
            elif proto_column_info.HasField(FEATURE_COLUMN_INFO):
                column_infos.append(
                    FeatureColumnInfo.from_proto(proto_column_info.feature_column_info)
                )
            else:
                raise ValueError(
                    f"Expected column_info to be a SourceDataColumnInfo or FeatureColumnInfo proto "
                    f"message. '{proto_column_info}' is of the wrong proto message type."
                )
        table_infos = [
            FeatureTableInfo.from_proto(proto_table_info)
            for proto_table_info in feature_spec_proto.input_tables
        ]
        return cls(
            column_infos=column_infos,
            table_infos=table_infos,
            workspace_id=feature_spec_proto.workspace_id,
            feature_store_client_version=feature_spec_proto.feature_store_client_version,
        )

    def to_proto(self):
        proto_feature_spec = ProtoFeatureSpec()
        for column_info in self.column_infos:
            ci = ProtoColumnInfo()
            # Using CopyFrom since both the SourceDataColumnInfo and FeatureColumnInfo create their
            # own protos
            if isinstance(column_info, SourceDataColumnInfo):
                ci.source_data_column_info.CopyFrom(column_info.to_proto())
            elif isinstance(column_info, FeatureColumnInfo):
                ci.feature_column_info.CopyFrom(column_info.to_proto())
            else:
                raise ValueError(
                    f"Expected column_info to be instances of SourceDataColumnInfo"
                    f" or FeatureColumnInfo. '{column_info}' is of the wrong type."
                )
            proto_feature_spec.input_columns.append(ci)
        for table_info in self.table_infos:
            proto_feature_spec.input_tables.append(table_info.to_proto())
        proto_feature_spec.serialization_version = self.SERIALIZATION_VERSION_NUMBER
        proto_feature_spec.workspace_id = self.workspace_id
        proto_feature_spec.feature_store_client_version = (
            self._feature_store_client_version
        )
        return proto_feature_spec

    @staticmethod
    def _dict_key_by_name(column_info):
        if SOURCE_DATA_COLUMN_INFO in column_info:
            source_data = column_info[SOURCE_DATA_COLUMN_INFO]
            source_name = source_data.pop(NAME)
            source_data[SOURCE] = TRAINING_DATA
            return {source_name: source_data}
        elif FEATURE_COLUMN_INFO in column_info:
            feature_data = column_info[FEATURE_COLUMN_INFO]
            feature_data[SOURCE] = FEATURE_STORE
            return {feature_data[OUTPUT_NAME]: feature_data}
        else:
            raise ValueError(
                f"Expected column_info to be keyed by '{SOURCE_DATA_COLUMN_INFO}' and "
                f"'{FEATURE_COLUMN_INFO}'. '{column_info}' has key '{list(column_info)[0]}'."
            )

    def _to_dict(self):
        """
        Convert FeatureSpec to a writeable YAML artifact. Uses MessageToDict to convert a
        FeatureSpec proto message to a dict, then modifies the dict to be keyed by name
        :return: dict with column infos keyed by column name
        """
        # In all newer feature store clients, the unique feature tables names in input columns should always match
        # feature table names in input tables.
        unique_tables_from_feature_column_infos = set(
            [
                feature_column_info.table_name
                for feature_column_info in self.feature_column_infos
            ]
        )
        unique_tables_from_table_infos = set(
            [table_info.table_name for table_info in self.table_infos]
        )
        if unique_tables_from_feature_column_infos != unique_tables_from_table_infos:
            raise Exception(
                "Internal Error: Feature table names from input_tables "
                "does not match feature table names from input_columns."
            )

        yaml_dict = MessageToDict(self.to_proto(), preserving_proto_field_name=True)
        yaml_dict[INPUT_COLUMNS] = [
            self._dict_key_by_name(column_info)
            for column_info in yaml_dict[INPUT_COLUMNS]
        ]

        if INPUT_TABLES in yaml_dict:
            yaml_dict[INPUT_TABLES] = [
                {table_info[TABLE_NAME]: table_info}
                for table_info in yaml_dict[INPUT_TABLES]
            ]
        # For readability, place SERIALIZATION_VERSION last in the dictionary.
        reordered_keys = [k for k in yaml_dict.keys() if k != SERIALIZATION_VERSION] + [
            SERIALIZATION_VERSION
        ]
        return {k: yaml_dict[k] for k in reordered_keys}

    def save(self, path: str):
        """
        Convert spec to a YAML artifact and store at given `path` location.
        :param path: Root path to where YAML artifact is expected to be stored.
        :return: None
        """
        # TODO(ML-15922): Migrate to use mlflow file_utils once `sort_keys` argument is supported
        fs_file_utils.write_yaml(
            path, self.FEATURE_ARTIFACT_FILE, self._to_dict(), sort_keys=False
        )

    @staticmethod
    def _dict_key_by_source(column_info):
        if len(column_info) != 1:
            raise ValueError(
                f"Expected column_info dictionary to only have one key, value pair. '{column_info}'"
                f" has length {len(column_info)}."
            )
        column_name, column_data = list(column_info.items())[0]
        if not column_data:
            raise ValueError(
                f"Expected values of '{column_name}' dictionary to be non-empty."
            )
        if SOURCE not in column_data:
            raise ValueError(
                f"Expected values of column_info dictionary to include the source. No source found "
                f"for '{column_name}'."
            )
        source = column_data.pop(SOURCE)
        if source == TRAINING_DATA:
            column_data[NAME] = column_name
            return {SOURCE_DATA_COLUMN_INFO: column_data}
        elif source == FEATURE_STORE:
            return {FEATURE_COLUMN_INFO: column_data}
        else:
            raise ValueError(
                f"Expected column_info to have source of '{TRAINING_DATA}' or '{FEATURE_STORE}'. "
                f"'{column_info}' has source of '{source}'."
            )

    @classmethod
    def _from_dict(cls, spec_dict):
        """
        Convert YAML artifact to FeatureSpec. Transforms YAML artifact to dict keyed by
        source_data_column_info or feature_column_info, such that ParseDict can convert the dict to
        a proto message, and from_proto can convert the proto message to a FeatureSpec object
        :return: :py:class:`~databricks.feature_store.entities.feature_spec.FeatureSpec`
        """
        if INPUT_COLUMNS not in spec_dict:
            raise ValueError(
                f"{INPUT_COLUMNS} must be a key in {cls.FEATURE_ARTIFACT_FILE}."
            )
        if not spec_dict[INPUT_COLUMNS]:
            raise ValueError(
                f"{INPUT_COLUMNS} in {cls.FEATURE_ARTIFACT_FILE} must be non-empty."
            )
        spec_dict[INPUT_COLUMNS] = [
            cls._dict_key_by_source(column_info)
            for column_info in spec_dict[INPUT_COLUMNS]
        ]
        # For feature_spec.yaml wrote by older version of the client and for feature_spec.yaml that only has
        # source data, table_infos field may not be present
        if INPUT_TABLES not in spec_dict:
            spec_dict[INPUT_TABLES] = []
        else:
            spec_dict[INPUT_TABLES] = [
                list(input_table.values())[0] for input_table in spec_dict[INPUT_TABLES]
            ]
            # table_name in all table_infos must be unique
            unique_table_names = set(
                [input_table[TABLE_NAME] for input_table in spec_dict[INPUT_TABLES]]
            )
            if len(spec_dict[INPUT_TABLES]) != len(unique_table_names):
                raise Exception(
                    "Internal Error: Expect all table_name in input_tables to be unique. Duplicate table_name found."
                )
        return cls.from_proto(
            ParseDict(spec_dict, ProtoFeatureSpec(), ignore_unknown_fields=True)
        )

    @classmethod
    def _read_file(cls, path: str):
        """
        Read the YAML artifact from a file path.
        """
        parent_dir, file = os.path.split(path)
        spec_dict = mlflow_file_utils.read_yaml(parent_dir, file)
        return cls._from_dict(spec_dict)

    @classmethod
    def load(cls, path: str):
        """
        Load the FeatureSpec YAML artifact in the provided root directory (at path/feature_spec.yaml).

        :param path: Root path to the YAML artifact. This can be a MLflow artifact path or file path.
        :return: :py:class:`~databricks.feature_store.entities.feature_spec.FeatureSpec`
        """
        # Create the full file path to the FeatureSpec.
        path = os.path.join(path, cls.FEATURE_ARTIFACT_FILE)

        if is_artifact_uri(path):
            with TempDir() as tmp_location:
                # Returns a file and not directory since the artifact_uri is a single file.
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=path, dst_path=tmp_location.path()
                )
                return FeatureSpec._read_file(local_path)
        else:
            return FeatureSpec._read_file(path)
