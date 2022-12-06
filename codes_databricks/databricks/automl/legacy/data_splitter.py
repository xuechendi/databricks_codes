import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, monotonically_increasing_id, percent_rank

from databricks.automl.legacy.problem_type import ClassificationTargetTypes
from databricks.automl.legacy.errors import UnsupportedDataError
from databricks.automl.legacy.stats import StatsCalculator


class DataSplitter(ABC):
    TIME_COL_PERCENT_PREFIX = "_automl_time_col_percent"
    INTERNAL_ID_PREFIX = "_automl_internal_id"
    SPLIT_COL_PREFIX = "_automl_split_col"

    @abstractmethod
    def split(self,
              df: DataFrame,
              target_col: str,
              ratios: List[float],
              time_col: Optional[str] = None,
              class_counts: Optional[Dict[ClassificationTargetTypes, int]] = None,
              seed: int = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
        pass

    def split_by_time_col(self, df: DataFrame, ratios: List[float],
                          time_col: str) -> Tuple[DataFrame, DataFrame, DataFrame]:

        time_col_percent = f"{self.TIME_COL_PERCENT_PREFIX}_{str(uuid.uuid4())[:4]}"

        # Sort by time order and split
        df = df.withColumn(time_col_percent,
                           percent_rank().over(Window.partitionBy().orderBy(time_col)))

        train_df = df.where(f"{time_col_percent} <= {ratios[0]}").drop(time_col_percent)

        val_df = df.where((df[time_col_percent] > ratios[0]) &
                          (df[time_col_percent] <= ratios[0] + ratios[1])).drop(time_col_percent)

        test_df = df.where(f"{time_col_percent} > {ratios[0] + ratios[1]}").drop(time_col_percent)

        return train_df, val_df, test_df


class ClassificationDataSplitter(DataSplitter):
    def _train_test_split(self,
                          df: DataFrame,
                          target_col: str,
                          train_ratio: float,
                          class_counts: Optional[Dict[ClassificationTargetTypes, int]] = None,
                          seed: int = None,
                          ensure_all_labels_present: bool = False):
        """
        Split dataframe to training and testing splits using stratified sampling on labels.
        :param df: full dataframe that is to be split
        :param target_col: target column with the label
        :param train_ratio: the ratio of the training data
        :param class_counts: the count of each label
        :return: dataframes training split, testing split
        """
        if ensure_all_labels_present:
            # Check that all labels have sufficient number of rows to avoid the training split
            # being empty after split by sampleBy(..)
            for clas, num_rows in class_counts.items():
                target_sample_size = StatsCalculator.PYSPARK_MIN_OUTPUT_ROWS_TO_INPUT_ROWS_MAP[
                    StatsCalculator.MIN_ROWS_PER_LABEL_AFTER_SPLIT]
                if num_rows * train_ratio < target_sample_size:
                    raise UnsupportedDataError(
                        f"Target label {clas} has only {num_rows} data points."
                        f"At least {target_sample_size / train_ratio} "
                        "data points are required per target label after train test split.")

        fractions = {target_class: train_ratio for target_class in class_counts.keys()}

        internal_id = f"{self.INTERNAL_ID_PREFIX}_{str(uuid.uuid4())[:4]}"

        df = df.withColumn(internal_id, monotonically_increasing_id())
        train_df = df.sampleBy(target_col, fractions, seed=seed).cache()
        test_df = df.join(
            train_df, how="left_anti", on=(df[internal_id] == train_df[internal_id])).cache()

        train_df = train_df.drop(col(internal_id))
        test_df = test_df.drop(col(internal_id))
        return train_df, test_df

    def split(self,
              df: DataFrame,
              target_col: str,
              ratios: List[float],
              time_col: Optional[str] = None,
              class_counts: Optional[Dict[ClassificationTargetTypes, int]] = None,
              seed: int = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
        assert len(ratios) == 3 and sum(ratios) == 1.0, \
            "Split function requires the ratios of the train/val/test data"

        if time_col:
            return self.split_by_time_col(df, ratios, time_col)

        train_df, split_rem_df = self._train_test_split(
            df=df,
            target_col=target_col,
            train_ratio=ratios[0],
            class_counts=class_counts,
            seed=seed,
            ensure_all_labels_present=True)

        rem_class_counts = StatsCalculator.get_class_counts(split_rem_df, target_col)

        val_df, test_df = self._train_test_split(
            df=split_rem_df,
            target_col=target_col,
            train_ratio=ratios[1] / (ratios[1] + ratios[2]),
            class_counts=rem_class_counts,
            seed=seed)

        return train_df, val_df, test_df


class RandomDataSplitter(DataSplitter):
    def split(self,
              df: DataFrame,
              target_col: str,
              ratios: List[float],
              time_col: Optional[str] = None,
              class_counts: Optional[Dict[ClassificationTargetTypes, int]] = None,
              seed: int = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
        assert len(ratios) == 3 and sum(ratios) == 1.0, \
            "Split function requires the ratios of the train/val/test data"

        if time_col:
            return self.split_by_time_col(df, ratios, time_col)

        train_df, val_df, test_df = df.randomSplit(ratios, seed=seed)
        return train_df, val_df, test_df
