import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pyspark.sql import DataFrame

from databricks.automl.internal.alerts import InferredPosLabelAlert, \
    TargetLabelImbalanceAlert, TargetLabelRatioAlert, InappropriateMetricForImbalanceAlert
from databricks.automl.internal.alerts.alert_manager import AlertManager
from databricks.automl.internal.confs import InternalConfs
from databricks.automl.internal.stats import IntermediateStats, StatsCalculator, InputStats
from databricks.automl.shared.const import ClassificationTargetTypes
from databricks.automl.shared.const import Metric
from databricks.automl.shared.errors import InvalidArgumentError

_logger = logging.getLogger(__name__)


@dataclass
class ClassificationInputStats(InputStats):
    """ The stats are used for logging and the rough sampling step. """
    class_counts: Dict[ClassificationTargetTypes, int]


@dataclass
class ClassificationIntermediateStats(IntermediateStats):
    class_counts: Dict[ClassificationTargetTypes, int]


class ClassificationStatsCalculator(StatsCalculator):
    # If below the threshold, the dataset is considered to be imbalanced
    IMBALANCE_RATIO_THRESHOLD = 0.1
    # If not imbalanced but below the threshold, show the ratio of the least to the most frequent label
    SHOW_RATIO_THRESHOLD = 0.3

    # 5 rows in total are required when we adopt the sample-and-split, so that we can
    # split the sampled data into train/val/test data for a minimum of 3/1/1 rows.
    MIN_ROWS_PER_LABEL_BEFORE_SPLIT = 5
    # Only 1 row is required after sampling when we adopt the split-and-sample workflow.
    MIN_ROWS_PER_LABEL_AFTER_SPLIT = 1
    # Number of rows required by pyspark.sampleBy(..) to make sure we at least get
    # 1 or 5 row(s) when we sample the dataset.
    # See the below notebook, where sampleBy(..) is called 1000 times to ensure enough rows are
    # left after sampleBy(..):
    # http://go/dogfood/?o=6051921418418893#notebook/3603418870987052/command/3603418870987064
    PYSPARK_MIN_OUTPUT_ROWS_TO_INPUT_ROWS_MAP = {1: 10, 5: 20}

    # Minimum number of rows per class required for AutoML to successfully run test-train-validation split
    MIN_ROWS_PER_CLASS = int(PYSPARK_MIN_OUTPUT_ROWS_TO_INPUT_ROWS_MAP[MIN_ROWS_PER_LABEL_AFTER_SPLIT] / 0.6) \
        if InternalConfs.ENABLE_TRAIN_TEST_SPLIT_DRIVER \
        else MIN_ROWS_PER_LABEL_BEFORE_SPLIT

    POSITIVE_LABEL_MAP = {
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

    def __init__(self,
                 target_col: str,
                 time_col: Optional[str],
                 run_id: str = None,
                 pos_label: Optional[ClassificationTargetTypes] = None,
                 metric: Metric = Metric.F1_SCORE):
        super().__init__(target_col, time_col, run_id)
        self._pos_label = pos_label
        self._metric = metric

    @classmethod
    def appropriate_metrics_for_imbalanced_data(cls) -> Iterable[Metric]:
        return [Metric.F1_SCORE, Metric.LOG_LOSS, Metric.PRECISION, Metric.ROC_AUC]

    def get_input_stats(self, dataset: DataFrame) -> ClassificationInputStats:
        """
        Calculate the stats used for InputDataStats logging and the rough sampling step.

        :param dataset: pyspark dataset
        :return: ClassificationInputStats
        """
        stats = super().get_input_stats(dataset)

        class_counts, num_low_class_count_rows = self._get_class_counts(dataset)

        input_stats = ClassificationInputStats(
            num_rows=stats.num_rows,
            num_invalid_rows=stats.num_invalid_rows + num_low_class_count_rows,
            num_target_nulls=stats.num_target_nulls,
            num_cols=stats.num_cols,
            num_string_cols=stats.num_string_cols,
            num_supported_cols=stats.num_supported_cols,
            class_counts=class_counts,
        )
        _logger.debug(f"InputStats calculated: {input_stats}")
        return input_stats

    def get_intermediate_stats(self,
                               dataset: DataFrame,
                               input_num_rows: int,
                               is_sampled: bool = False) -> IntermediateStats:
        """
        Make one pass on the pyspark dataset and calculate the total number or rows,
        average length of string columns, and the approximate cardinality of string columns
        Also include information about the dataset schema

        :param dataset: pyspark dataset
        :param dataset: pyspark dataset
        :return: IntermediateStats
        """
        stats = super().get_intermediate_stats(
            dataset, input_num_rows=input_num_rows, is_sampled=is_sampled)

        class_counts, num_low_class_count_rows = self._get_class_counts(dataset)

        pre_sampling_stats = ClassificationIntermediateStats(
            num_rows=stats.num_rows,
            schema_map=stats.schema_map,
            str_columns=stats.str_columns,
            columns=stats.columns,
            feature_schema=stats.feature_schema,
            supported_cols=stats.supported_cols,
            feature_metadata=stats.feature_metadata,
            class_counts=class_counts,
            num_invalid_rows=stats.num_invalid_rows)
        _logger.debug(f"DataStats calculated before precise sampling: {pre_sampling_stats}")
        return pre_sampling_stats

    def _get_class_counts(self,
                          dataset: DataFrame) -> Tuple[Dict[ClassificationTargetTypes, int], int]:
        # Calculate the class counts and also calculate the number of
        # rows that have less than the min required class counts
        class_counts = self.get_class_counts(dataset)
        num_low_class_count_rows = sum([
            ct for c, ct in class_counts.items()
            if ct < ClassificationStatsCalculator.MIN_ROWS_PER_CLASS and c is not None
        ])
        return class_counts, num_low_class_count_rows

    def validate_intermediate_stats(
            self, intermediate_stats: IntermediateStats, alert_manager: AlertManager
    ) -> Tuple[bool, Optional[List[float]], Optional[ClassificationTargetTypes]]:
        """
        Validate the pre-sampling stats for classification problem. Besides the validations for
        unsupported columns, it also validates positive labels if provided and check whether we
        should balance the dataset.
        :param intermediate_stats: Result for pre sampling.
        :param alert_manager: AlertManager used to pass warnings to the user.
        :return: indicator whether we should balance data, list of target_label_ratios, and
            processed positive label
        """
        self._validate_unsupported_columns(intermediate_stats, alert_manager)

        class_counts = intermediate_stats.class_counts

        # Validate positive label
        if self._pos_label:
            self._validate_and_cast_pos_label(list(intermediate_stats.class_counts.keys()))
        elif len(class_counts) == 2:
            # Infer pos_label for binary classification if it isn't specified
            self._pos_label = self._infer_pos_label(class_counts)
            alert_manager.record(InferredPosLabelAlert(self._pos_label))
            _logger.debug(f"Positive label inferred as: {self._pos_label}")

        should_balance, target_label_ratios = self._should_balance(intermediate_stats.class_counts,
                                                                   alert_manager)

        return should_balance, target_label_ratios, self._pos_label

    def get_class_counts(self, dataset: DataFrame) -> Dict[ClassificationTargetTypes, int]:
        """
        Get the number of rows for each class in a classification dataset
        When the cardinality of the target column is high, this query is very slow!
        :return: dictionary where each key is a class and each value is the number of rows of that class
        """
        class_counts_df = dataset.groupBy(self._target_col).count()
        class_counts = {row[self._target_col]: row["count"] for row in class_counts_df.collect()}
        return class_counts

    def _validate_and_cast_pos_label(self, labels: List[str]) -> None:
        # Check if pos_label is passed for multi-class
        num_labels = len(labels)
        if num_labels != 2:
            raise InvalidArgumentError(
                f"AutoML detected {num_labels} target classes in the dataset. "
                "Argument `pos_label` can only be used with a binary dataset that has exactly 2 classes."
            )

        pos_label_type = type(self._pos_label)
        target_label_type = type(labels[0])

        # Cast pos_label to the same type of labels for comparison. The goal is to make the pos_label valid
        # if the string representations of the pos_label and one real target label are the same, which is
        # only possible for between str and int or between str and bool.
        if pos_label_type != target_label_type:
            if target_label_type == str or (target_label_type == int and pos_label_type == str):
                try:
                    self._pos_label = target_label_type(self._pos_label)
                except ValueError:
                    pass
            elif target_label_type == bool:
                # bool() returns True on anything except empty string, False, 0 or None
                # So we handle the casting to bool differently than other cases
                if self._pos_label == "True":
                    self._pos_label = True
                elif self._pos_label == "False":
                    self._pos_label = False

        # Check if label is invalid
        if self._pos_label not in labels:
            raise InvalidArgumentError(
                f"The specified pos_label={self._pos_label} (type {pos_label_type}) does not belong to "
                f"the labels found in the dataset {labels} (type {target_label_type}).")

    @staticmethod
    def _infer_pos_label(
            class_counts: Dict[ClassificationTargetTypes, int]) -> ClassificationTargetTypes:
        assert len(class_counts) == 2, \
            f"Should only infer pos_label when there are 2 classes, now there are {len(class_counts)} classes"

        # Infer pos_label based on common positive label names
        class_labels = set(class_counts.keys())
        for target_labels, pos_label in ClassificationStatsCalculator.POSITIVE_LABEL_MAP.items():
            if set(target_labels) == class_labels:
                return pos_label

        # Otherwise, sort the classes by their count and return the class with the smallest count
        return sorted(list(class_counts.items()), key=lambda x: x[1])[0][0]

    def _should_balance(self, class_counts: Dict[ClassificationTargetTypes, int],
                        alert_manager: AlertManager) -> Tuple[bool, Optional[List[float]]]:
        """
        Determine whether the dataset is imbalanced, which satisfy both:
          1. has at least 2 classes
          2. the ratio of the least frequent label to the most frequent label is less than IMBALANCE_RATIO_THRESHOLD
        and send alerts of different severity based on the imbalance ratio.

        :return: Tuple, the first element is True if the dataset is imbalanced and an appropriate metric is chosen;
        the second element is list of ratios between the number of samples of each class and the largest class.
        """
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
            if self._metric in self.appropriate_metrics_for_imbalanced_data():
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
                        metric=self._metric.short_name,
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
