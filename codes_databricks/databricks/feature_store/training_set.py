from databricks.feature_store.entities.feature_spec import FeatureSpec
from databricks.feature_store.entities.feature_table import FeatureTable
from databricks.feature_store.utils.feature_lookup_utils import join_feature_data
from pyspark.sql import DataFrame

from typing import List, Dict


class TrainingSet:
    """
    Class that defines :obj:`TrainingSet` objects.

    .. note::

       The :class:`TrainingSet` constructor should not be called directly. Instead,
       call :meth:`FeatureStoreClient.create_training_set <databricks.feature_store.client.FeatureStoreClient.create_training_set>`.
    """

    def __init__(
        self,
        feature_spec: FeatureSpec,
        df: DataFrame,
        labels: List[str],
        feature_table_metadata_map: Dict[str, FeatureTable],
        feature_table_data_map: Dict[str, DataFrame],
    ):
        """Initialize a :obj:`TrainingSet` object."""
        assert isinstance(
            labels, list
        ), f"Expected type `list` for argument `labels`. Got '{labels}' with type '{type(labels)}'."

        self._feature_spec = feature_spec
        self._df = df
        self._labels = labels
        self._feature_table_metadata_map = feature_table_metadata_map
        self._feature_table_data_map = feature_table_data_map

    @property
    def feature_spec(self) -> FeatureSpec:
        """Define a feature spec."""
        return self._feature_spec

    def load_df(self) -> DataFrame:
        """
        Load a :class:`DataFrame <pyspark.sql.DataFrame>`.

        Return a :class:`DataFrame <pyspark.sql.DataFrame>` for training.

        The returned :class:`DataFrame <pyspark.sql.DataFrame>` has columns specified
        in the ``feature_spec`` and ``labels`` parameters provided
        in :meth:`FeatureStoreClient.create_training_set <databricks.feature_store.client.FeatureStoreClient.create_training_set>`.

        :return:
           A :class:`DataFrame <pyspark.sql.DataFrame>` for training
        """
        return join_feature_data(
            feature_spec=self._feature_spec,
            df=self._df,
            extra_columns=self._labels,
            feature_table_metadata_map=self._feature_table_metadata_map,
            feature_table_data_map=self._feature_table_data_map,
        )
