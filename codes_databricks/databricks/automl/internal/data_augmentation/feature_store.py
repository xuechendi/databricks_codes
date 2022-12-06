import functools
import inspect
from typing import Any, Dict, List, Set, Tuple

from pyspark.sql import DataFrame

from databricks.automl.shared.errors import FeatureStoreError
from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup
from databricks.feature_store.entities.feature_spec import FeatureSpec


def error_wrapper(func):
    """
    Decorator to wrap a function's raised Exceptions with FeatureStoreError.
    Forwards all of a function call's arguments and return values.
    """

    @functools.wraps(func)
    def wrapper(*a, **kw):
        try:
            return func(*a, **kw)
        except Exception as e:
            raise FeatureStoreError(e) from e

    return wrapper


def decorate_all_methods(decorator):
    """
    Apply a decorator to all functions of a class that is passed in.
    Iterates through all members of the class and applies the decorator to all functions.
    """

    def apply_decorator(cls):
        for k, f in cls.__dict__.items():
            if inspect.isfunction(f):
                setattr(cls, k, decorator(f))
        return cls

    return apply_decorator


"""
Wraps all methods of the FeatureStoreClient with error_wrapper. AutoML code should not use
FeatureStoreClient directly, but should instead use AutoMLFeatureStoreClient. This is to
help categorize errors in our instrumentation.
"""
AutoMLFeatureStoreClient = decorate_all_methods(error_wrapper)(FeatureStoreClient)


class FeatureStoreJoiner:
    """
    Errors raised by FeatureStoreJoiner are wrapped with FeatureStoreError.
    """

    def __init__(self, feature_store_lookups: List[Dict]):
        """
        :param feature_store_lookups: list of dicts with the specification of features to join
        """
        self._feature_store_lookups = feature_store_lookups
        self._fs_client = AutoMLFeatureStoreClient()

    def get_num_features(self) -> int:
        """
        Get total number of features in given feature store lookups.
        :return: num_features
        """
        total_num_features = 0
        for raw_lookup in self._feature_store_lookups:
            feature_table = self._fs_client.get_table(raw_lookup["table_name"])
            num_primary_keys = 1 if isinstance(feature_table.primary_keys, str) else len(
                feature_table.primary_keys)
            num_timestamp_keys = 1 if isinstance(feature_table.timestamp_keys, str) \
                else len(feature_table.timestamp_keys)
            total_num_features += len(
                feature_table.features) - num_primary_keys - num_timestamp_keys
        return total_num_features

    def join_features(self, dataset: DataFrame, target_col: str
                      ) -> Tuple[DataFrame, FeatureSpec, Dict[str, Dict[str, str]], Set[str]]:
        """
        Join Feature Store features with the dataset.
        :param dataset: spark DataFrame
        :param target_col: target column name
        :return: (joined_dataset, feature_spec)
            joined_dataset: spark DataFrame include the input dataset and joined features
            feature_spec: FeatureSpec which is necessary for logging model trained with Feature Store features.
                          This will be saved together with the data.
            column_rename_map: rename dict of each table
            joined_cols: a set of column names joined from Feature Store
        """
        original_cols_set = set(dataset.columns)
        feature_lookups, column_rename_map = self._construct_feature_lookups(original_cols_set)
        training_set = self._fs_client.create_training_set(
            dataset, feature_lookups=feature_lookups, label=target_col)
        try:
            joined_dataset = training_set.load_df()
        except Exception as e:
            raise FeatureStoreError(e) from e
        feature_spec = training_set.feature_spec
        joined_cols = set(joined_dataset.columns) - original_cols_set
        return joined_dataset, feature_spec, column_rename_map, joined_cols

    def _construct_feature_lookups(
            self, column_set: Set[str]) -> Tuple[List[FeatureLookup], Dict[str, Dict[str, str]]]:
        feature_lookups = []
        column_name_map = {}
        for raw_lookup in self._feature_store_lookups:
            column_set, rename_outputs = self._construct_column_map(raw_lookup, column_set)
            lookup = FeatureLookup(
                table_name=raw_lookup["table_name"],
                lookup_key=raw_lookup["lookup_key"],
                rename_outputs=rename_outputs,
                timestamp_lookup_key=raw_lookup.get("timestamp_lookup_key", None))
            feature_lookups.append(lookup)
            column_name_map[raw_lookup["table_name"]] = rename_outputs
        return feature_lookups, column_name_map

    def _construct_column_map(self, feature_lookup: Dict[str, Any],
                              column_set: Set[str]) -> Tuple[Set[str], Dict[str, str]]:
        table_name = feature_lookup["table_name"]
        feature_table = self._fs_client.get_table(table_name)
        table_name_underscore = table_name.replace(".", "_")
        features = set(feature_table.features)
        lookup_keys = feature_table.primary_keys
        if type(lookup_keys) == str:
            lookup_keys = [lookup_keys]
        timestamp_keys = feature_table.timestamp_keys
        features = features - set(lookup_keys)
        features = features - set(timestamp_keys)
        rename_outputs = {}
        for feature in features:
            if feature in column_set:
                new_feature_name = f"{table_name_underscore}_{feature}"
                idx = 1
                while new_feature_name in column_set:
                    new_feature_name = f"{new_feature_name}_{idx}"
                    idx = idx + 1
                rename_outputs[feature] = new_feature_name
                column_set.add(new_feature_name)
            else:
                column_set.add(feature)

        return column_set, rename_outputs
