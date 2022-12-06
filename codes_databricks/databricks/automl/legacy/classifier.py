import logging
from itertools import chain
from typing import Dict, List, Iterable, Optional, Tuple

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, create_map, lit
from pyspark.sql.types import DataType, BooleanType, StringType

from databricks.automl.legacy.alerts import UnableToSampleWithoutSkewAlert, TargetLabelInsufficientDataAlert, \
    InferredPosLabelAlert, TargetLabelImbalanceAlert, TargetLabelRatioAlert, InappropriateMetricForImbalanceAlert
from databricks.automl.legacy.alerts.alert_manager import AlertManager
from databricks.automl.legacy.classification_planner import SklearnLogisticRegressionTrialPlanner, \
    SklearnDecisionTreeTrialPlanner, \
    SklearnRandomForestTrialPlanner, SklearnXGBoostTrialPlanner, SklearnLGBMTrialPlanner
from databricks.automl.legacy.data_splitter import DataSplitter, ClassificationDataSplitter
from databricks.automl.legacy.errors import InvalidArgumentError, UnsupportedDataError
from databricks.automl.legacy.planner import TrialPlanner
from databricks.automl.legacy.preprocess import SupervisedLearnerDataPreprocessor
from databricks.automl.legacy.problem_type import ClassificationTargetTypes, ProblemType, Metric
from databricks.automl.legacy.stats import PreSamplingStats, StatsCalculator
from databricks.automl.legacy.supervised_learner import SupervisedLearner

_logger = logging.getLogger(__name__)


class Classifier(SupervisedLearner):
    """
    Implementation of databricks.automl.classify().
    """
    # If below the threshold, the dataset is considered to be imbalanced
    IMBALANCE_RATIO_THRESHOLD = 0.1
    # If not imbalanced but below the threshold, show the ratio of the least to the most frequent label
    SHOW_RATIO_THRESHOLD = 0.3

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.CLASSIFICATION

    @property
    def default_metric(self) -> Metric:
        return Metric.F1_SCORE

    @property
    def splitter(self) -> DataSplitter:
        return ClassificationDataSplitter()

    @classmethod
    def supported_metrics(cls) -> Iterable[Metric]:
        return [Metric.F1_SCORE, Metric.LOG_LOSS, Metric.PRECISION, Metric.ACCURACY, Metric.ROC_AUC]

    @classmethod
    def appropriate_metrics_for_imbalanced_data(cls) -> Iterable[Metric]:
        return [Metric.F1_SCORE, Metric.LOG_LOSS, Metric.PRECISION, Metric.ROC_AUC]

    @classmethod
    def _get_supported_target_types(cls) -> List[DataType]:
        """
        This should be consistent with ClassificationTargetTypes in problem_type.py
        """
        return SupervisedLearnerDataPreprocessor.INTEGER_TYPES + (BooleanType, StringType)

    def _get_planners(self) -> List[TrialPlanner]:
        return [
            SklearnDecisionTreeTrialPlanner, SklearnLogisticRegressionTrialPlanner,
            SklearnRandomForestTrialPlanner, SklearnXGBoostTrialPlanner, SklearnLGBMTrialPlanner
        ]

    def _drop_invalid_rows(self, dataset: DataFrame, target_col: str,
                           dataset_stats: PreSamplingStats,
                           alert_manager: AlertManager) -> DataFrame:
        """
        Drop invalid rows.
        Additionally drop the rows that don't have enough target column labels

        :param dataset: pyspark dataset
        :param target_col: target column with the label
        :param dataset_stats: pre sampling stats calculated over the dataset

        :return: dataset with the transformation to drop invalid rows
        """
        dataset = super()._drop_invalid_rows(dataset, target_col, dataset_stats, alert_manager)

        # if a target class does not have enough data, drop the rows with that class
        invalid_classes = [
            c for c, count in dataset_stats.class_counts.items()
            if count < StatsCalculator.MIN_ROWS_PER_CLASS and c is not None
        ]
        if invalid_classes:
            alert_manager.record(TargetLabelInsufficientDataAlert(invalid_classes))
            dataset = dataset.where(~col(target_col).isin(invalid_classes))
        return dataset

    def _sample(self,
                dataset: DataFrame,
                fraction: float,
                target_col: str,
                dataset_stats: PreSamplingStats,
                alert_manager: AlertManager,
                min_rows_to_ensure: int = 5) -> DataFrame:
        """
        Stratified sampling to retain the same proportions among labels, and ensure that the labels are present
        after sampling.
        :param dataset: the dataframe that needs to be sampled
        :param fraction: the fraction to sample
        :param target_col: target column with labels
        :param dataset_stats: pre-sampling stats of the dataset
        :param min_rows_to_ensure: minimum rows to be present for each label after sampling
        :return sampled dataframe
        """
        fractions = self._get_stratified_fractions(fraction, dataset_stats, alert_manager,
                                                   min_rows_to_ensure)
        seed = np.random.randint(1e9)
        return dataset.sampleBy(target_col, fractions, seed)

    def _sample_and_balance(self, dataset: DataFrame, fraction: float, target_col: str,
                            dataset_stats: PreSamplingStats, sample_weight_col: str) -> DataFrame:
        """
        Sample the data with the ratio of each label as close as possible, and compute the weight of each sample.
        The sample weight of a sample is 1 / the fraction of samples remaining for the class it belongs to, which
        serves the purpose of upweighting the downsampled classes.
        :param dataset: the dataframe that needs to be sampled
        :param fraction: the fraction to sample
        :param target_col: target column with labels
        :param dataset_stats: pre-sampling stats of the dataset
        :param sample_weight_col: column name of the sample weight that should be added to the sampled dataframe
        :return: sampled dataframe with a new sample weight column
        """
        fractions = self._get_balanced_fractions(fraction, dataset_stats)

        seed = np.random.randint(1e9)
        sampled_data = dataset.sampleBy(target_col, fractions, seed)

        class_weights = {k: 1 / v for k, v in fractions.items()}
        sample_weight_expr = create_map([lit(x) for x in chain(*class_weights.items())])

        sampled_data = sampled_data.withColumn(sample_weight_col,
                                               sample_weight_expr[col(target_col)])
        return sampled_data

    def _get_stratified_fractions(
            self,
            fraction: float,
            dataset_stats: PreSamplingStats,
            alert_manager: AlertManager,
            min_rows_to_ensure: Optional[int] = None,
    ) -> Dict[ClassificationTargetTypes, float]:
        """
        Create a dictionary that maps each class to the sampling fraction of that class. Subject to two constraints:
        1. Target minimum number of rows for each class in the sample is MIN_ROWS_PER_CLASS
           (but in reality may be less because sampling APIs don't return an exact number of rows, and some classes
           may have less than MIN_ROWS_PER_CLASS rows)
        2. Total number of sample rows divided by total number of original rows is equal to sampling fraction
        :param fraction: sampling fraction
        :param dataset_stats: pre sampling dataset stats
        :param min_rows_to_ensure: number of rows per label that should be present
        :return: dictionary where keys are classes and values are the sampling fractions for each class
        """
        total_num_rows = dataset_stats.num_rows - dataset_stats.num_invalid_rows
        class_counts = dataset_stats.class_counts

        fractions = {}
        sampled_num_rows = 0
        remaining_num_rows = 0

        # Number of rows for pyspark.sampleBy(..) to aim to yield
        if min_rows_to_ensure not in StatsCalculator.PYSPARK_MIN_OUTPUT_ROWS_TO_INPUT_ROWS_MAP:
            min_rows_to_ensure = StatsCalculator.MIN_ROWS_PER_LABEL_BEFORE_SPLIT
        target_sample_size = StatsCalculator.PYSPARK_MIN_OUTPUT_ROWS_TO_INPUT_ROWS_MAP.get(
            min_rows_to_ensure)

        # Determine the sampling fractions for rare classes
        for clas, num_rows in class_counts.items():
            # If a class does not have enough rows or is Null, it will not be present in the
            # dataset that's being sampled (since we drop the invalid rows before sampling)
            if clas is None or num_rows < min_rows_to_ensure:
                fractions[clas] = 0.0
            # If a class after sampling to the fraction is going to have less than target
            # number of rows, include all the rows (if fewer than target sample size) or enough rows to make
            # num_rows * fraction = at least target sample size
            elif num_rows * fraction < target_sample_size:
                if num_rows < target_sample_size:
                    fractions[clas] = 1.0
                    sampled_num_rows += num_rows
                else:
                    fractions[clas] = target_sample_size / num_rows
                    sampled_num_rows += target_sample_size
            else:
                remaining_num_rows += num_rows

        num_rare_classes = len(fractions)
        if remaining_num_rows > 0:
            # Determine the sampling fraction for common classes
            common_class_fraction = (fraction * total_num_rows - sampled_num_rows) \
                                    / remaining_num_rows
            for clas in class_counts:
                if clas not in fractions:
                    if common_class_fraction * class_counts[clas] < target_sample_size:
                        alert_manager.record(UnableToSampleWithoutSkewAlert())
                        raise UnsupportedDataError(
                            "Unable to sample the dataset without skewing the target label distribution. "
                            f"Dataset has {len(class_counts.keys())} target labels "
                            f"out of which {num_rare_classes} labels don't have enough data points. "
                            "Please pass a dataset with enough data points for each target label.")
                    fractions[clas] = common_class_fraction
        return fractions

    def _get_balanced_fractions(
            self,
            fraction: float,
            dataset_stats: PreSamplingStats,
    ) -> Dict[ClassificationTargetTypes, float]:
        """
        Create a dictionary that maps each class to the balanced sampling fraction of that class.
        This is equivalent to sampling with the given fraction, and trying to enable each label to have as
        similar number of rows as possible.
        :param fraction: sampling fraction
        :param dataset_stats: pre sampling dataset stats
        :return: dictionary where keys are classes and values are the sampling fractions for each class
        """
        fractions = {}

        total_num_rows = dataset_stats.num_rows - dataset_stats.num_invalid_rows
        class_counts = dataset_stats.class_counts

        num_labels_to_sample = len(class_counts)
        # Ideally the sampled rows consist of the given fraction of total rows,
        # but also ensure that minimum rows for each label are selected
        num_rows_to_fill = max(
            int(total_num_rows * fraction),
            int(StatsCalculator.PYSPARK_MIN_OUTPUT_ROWS_TO_INPUT_ROWS_MAP[
                StatsCalculator.MIN_ROWS_PER_LABEL_AFTER_SPLIT] * num_labels_to_sample))

        # Select rows from the labels with the least rows first
        for label, num_rows in sorted(class_counts.items(), key=lambda x: x[1]):
            # Try to take an equal number of rows from each remaining label
            ideal_num_rows_per_label = int(num_rows_to_fill / num_labels_to_sample)

            if num_rows <= ideal_num_rows_per_label:
                fractions[label] = 1.0
                num_rows_to_fill -= num_rows
            else:
                fractions[label] = ideal_num_rows_per_label / num_rows
                num_rows_to_fill -= ideal_num_rows_per_label
            num_labels_to_sample -= 1

        return fractions

    def _process_pos_label(self,
                           pre_sampling_stats: PreSamplingStats,
                           target_col: str,
                           alert_manager: AlertManager,
                           pos_label: Optional[ClassificationTargetTypes] = None
                           ) -> Optional[ClassificationTargetTypes]:
        class_counts = pre_sampling_stats.class_counts

        if pos_label:
            pos_label = self._validate_and_cast_pos_label(pre_sampling_stats, pos_label)
        elif len(class_counts) == 2:
            # Infer pos_label for binary classification if it isn't specified
            pos_label = self._infer_pos_label(class_counts)
            alert_manager.record(InferredPosLabelAlert(pos_label))
            _logger.debug(f"Positive label inferred as: {pos_label}")

        return pos_label

    def _validate_and_cast_pos_label(
            self, pre_sampling_stats: PreSamplingStats,
            pos_label: ClassificationTargetTypes) -> ClassificationTargetTypes:

        labels = list(pre_sampling_stats.class_counts.keys())

        # Check if pos_label is passed for multi-class
        num_labels = len(labels)
        if num_labels != 2:
            raise InvalidArgumentError(
                f"AutoML detected {num_labels} target classes in the dataset. "
                "Argument `pos_label` can only be used with a binary dataset that has exactly 2 classes."
            )

        pos_label_type = type(pos_label)
        target_label_type = type(labels[0])

        # Cast pos_label to the same type of labels for comparison. The goal is to make the pos_label valid
        # if the string representations of the pos_label and one real target label are the same, which is
        # only possible for between str and int or between str and bool.
        if pos_label_type != target_label_type:
            if target_label_type == str or (target_label_type == int and pos_label_type == str):
                try:
                    pos_label = target_label_type(pos_label)
                except ValueError:
                    pass
            elif target_label_type == bool:
                # bool() returns True on anything except empty string, False, 0 or None
                # So we handle the casting to bool differently than other cases
                if pos_label == "True":
                    pos_label = True
                elif pos_label == "False":
                    pos_label = False

        # Check if label is invalid
        if pos_label not in labels:
            raise InvalidArgumentError(
                f"The specified pos_label={pos_label} (type {pos_label_type}) does not belong to "
                f"the labels found in the dataset {labels} (type {target_label_type}).")

        return pos_label

    @staticmethod
    def _infer_pos_label(
            class_counts: Dict[ClassificationTargetTypes, int]) -> ClassificationTargetTypes:
        assert len(class_counts) == 2, \
            f"Should only infer pos_label when there are 2 classes, now there are {len(class_counts)} classes"

        # Infer pos_label based on common positive label names
        positive_label_map = {
            (True, False): True,
            (1, 0): 1,
            ("True", "False"): "True",
            ("1", "0"): "1",
            ("y", "n"): "y",
            ("yes", "no"): "yes",
            ("true", "false"): "true",
            ("pos", "neg"): "pos",
            ("positive", "negative"): "positive"
        }

        class_labels = set(class_counts.keys())
        for target_labels, pos_label in positive_label_map.items():
            if set(target_labels) == class_labels:
                return pos_label

        # Otherwise, sort the classes by their count and return the class with the smallest count
        return sorted(list(class_counts.items()), key=lambda x: x[1])[0][0]

    def _should_balance(self,
                        pre_sampling_stats: PreSamplingStats,
                        alert_manager: AlertManager,
                        metric: Metric = Metric.F1_SCORE) -> Tuple[bool, Optional[List[float]]]:
        """
        Determine whether the dataset is imbalanced, which satisfy both:
          1. has at least 2 classes
          2. the ratio of the least frequent label to the most frequent label is less than IMBALANCE_RATIO_THRESHOLD
        and send alerts of different severity based on the imbalance ratio.

        :return: Tuple, the first element is True if the dataset is imbalanced and an appropriate metric is chosen;
        the second element is list of ratios between the number of samples of each class and the largest class.
        """
        class_counts = pre_sampling_stats.class_counts
        if len(class_counts) < 2:
            return False, None

        most_frequent_label = max(class_counts, key=class_counts.get)
        least_frequent_label = min(class_counts, key=class_counts.get)
        ratio = class_counts.get(least_frequent_label) / class_counts.get(most_frequent_label)

        _logger.debug(f"Smallest imbalance ratio: {ratio}")

        # Log imbalance stats
        target_label_ratios = [
            num_rows / class_counts[most_frequent_label] for num_rows in class_counts.values()
        ]

        if ratio < self.IMBALANCE_RATIO_THRESHOLD:
            if metric in self.appropriate_metrics_for_imbalanced_data():
                alert_manager.record(
                    TargetLabelImbalanceAlert(
                        most_frequent_label=most_frequent_label,
                        least_frequent_label=least_frequent_label,
                        ratio=ratio))
                return True, target_label_ratios
            else:
                alert_manager.record(
                    InappropriateMetricForImbalanceAlert(
                        most_frequent_label=most_frequent_label,
                        least_frequent_label=least_frequent_label,
                        ratio=ratio,
                        metric=metric.short_name,
                        appropriate_metric=self.appropriate_metrics_for_imbalanced_data()
                        [0].short_name))
                return False, target_label_ratios

        if ratio < self.SHOW_RATIO_THRESHOLD:
            alert_manager.record(
                TargetLabelRatioAlert(
                    most_frequent_label=most_frequent_label,
                    least_frequent_label=least_frequent_label,
                    ratio=ratio))

        return False, target_label_ratios
