from databricks.feature_store.entities.online_feature_table import (
    AbstractOnlineFeatureTable,
    OnlineFeatureTable,
    OnlineFeatureTableForSageMakerServing,
)
from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject
from databricks.feature_store.protos.feature_store_serving_pb2 import (
    FeatureTablesForServing as ProtoFeatureTablesForServing,
    FeatureTablesForSageMakerServing as ProtoFeatureTablesForSageMakerServing,
)

from typing import List
import abc
import os


class AbstractFeatureTablesForServing(_FeatureStoreObject):
    def __init__(self, online_feature_tables: List[AbstractOnlineFeatureTable]):
        self._online_feature_tables = online_feature_tables

    @property
    def online_feature_tables(self):
        return self._online_feature_tables

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str):
        """
        Loads a binary serialized ProtoFeatureTablesForServing protocol buffer.

        :param path: Root path to the binary file.
        :return: :py:class:`~databricks.feature_store.entities.feature_tables_for_serving.FeatureTablesForServing`
        """
        pass


class FeatureTablesForServing(AbstractFeatureTablesForServing):
    DATA_FILE = "feature_tables_for_serving.dat"

    def __init__(self, online_feature_tables: List[OnlineFeatureTable]):
        super().__init__(online_feature_tables)

    @classmethod
    def from_proto(cls, the_proto: ProtoFeatureTablesForServing):
        online_fts = [
            OnlineFeatureTable.from_proto(online_table)
            for online_table in the_proto.online_tables
        ]
        return cls(online_feature_tables=online_fts)

    @classmethod
    def load(cls, path: str):
        """
        Loads a binary serialized ProtoFeatureTablesForServing protocol buffer.

        :param path: Root path to the binary file.
        :return: :py:class:`~databricks.feature_store.entities.feature_tables_for_serving.FeatureTablesForServing`
        """
        proto = ProtoFeatureTablesForServing()
        with open(os.path.join(path, cls.DATA_FILE), "rb") as f:
            proto.ParseFromString(f.read())
        return cls.from_proto(proto)


class FeatureTablesForSageMakerServing(AbstractFeatureTablesForServing):
    DATA_FILE = "feature_tables_for_sagemaker_serving.dat"

    def __init__(
        self, online_feature_tables: List[OnlineFeatureTableForSageMakerServing]
    ):
        super().__init__(online_feature_tables)

    def to_proto(self) -> ProtoFeatureTablesForSageMakerServing:
        proto = ProtoFeatureTablesForSageMakerServing()
        for olft in self.online_feature_tables:
            proto.online_tables.append(olft.to_proto())
        return proto

    @classmethod
    def from_proto(cls, the_proto: ProtoFeatureTablesForSageMakerServing):
        online_fts = [
            OnlineFeatureTableForSageMakerServing.from_proto(online_table)
            for online_table in the_proto.online_tables
        ]
        return cls(online_fts)

    @classmethod
    def load(cls, path: str):
        """
        Loads a binary serialized FeatureTablesForSageMakerServing protocol buffer.

        :param path: Root path to the binary file.
        :return: :py:class:`~databricks.feature_store.entities.feature_tables_for_serving.FeatureTablesForSageMakerServing`
        """
        proto = ProtoFeatureTablesForSageMakerServing()
        with open(os.path.join(path, cls.DATA_FILE), "rb") as f:
            proto.ParseFromString(f.read())
        return cls.from_proto(proto)
