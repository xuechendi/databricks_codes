from enum import Enum

TAG_PREFIX = "_databricks_automl"


class Tag(str, Enum):
    BASE = TAG_PREFIX

    STATE = f"{TAG_PREFIX}.state"
    START_TIME = f"{TAG_PREFIX}.start_time"
    END_TIME = f"{TAG_PREFIX}.end_time"
    ERROR_MESSAGE = f"{TAG_PREFIX}.error_message"
    EXPLORATION_NOTEBOOK_ID = f"{TAG_PREFIX}.exploration_notebook_id"
    BEST_TRIAL_NOTEBOOK_ID = f"{TAG_PREFIX}.best_trial_notebook_id"
    SAMPLE_FRACTION = f"{TAG_PREFIX}.sample_fraction"
    SOURCE_GUI = f"{TAG_PREFIX}.source_gui"
    OUTPUT_TABLE_NAME = f"{TAG_PREFIX}.output_table_name"
    IMPUTERS = f"{TAG_PREFIX}.imputers"
    FEATURE_STORE_LOOKUPS = f"${TAG_PREFIX}.feature_store_lookups"

    # Tags that are set in context.py set_experiment_init
    PROBLEM_TYPE = f"{TAG_PREFIX}.problem_type"
    TARGET_COL = f"{TAG_PREFIX}.target_col"
    DATA_DIR = f"{TAG_PREFIX}.data_dir"
    TIMEOUT_MINUTES = f"{TAG_PREFIX}.timeout_minutes"
    MAX_TRIALS = f"{TAG_PREFIX}.max_trials"
    EVALUATION_METRIC = f"{TAG_PREFIX}.evaluation_metric"
    EVALUATION_METRIC_ASC = f"{TAG_PREFIX}.evaluation_metric_order_by_asc"
    JOB_RUN_ID = f"{TAG_PREFIX}.job_run_id"

    # Alert tags should be in alphabetical order
    ALERT_ALL_ROWS_INVALID = f"{TAG_PREFIX}.alerts.all_rows_invalid"
    ALERT_ARRAY_NOT_NUMERICAL = f"{TAG_PREFIX}.alerts.array_not_numerical"
    ALERT_ARRAY_NOT_SAME_LENGTH = f"{TAG_PREFIX}.alerts.array_not_same_length"
    ALERT_CONSTANT_COLUMNS = f"{TAG_PREFIX}.alerts.const_cols"
    ALERT_DATA_EXPLORATION_FAIL = f"{TAG_PREFIX}.alerts.data_exploration_fail"
    ALERT_DATA_EXPLORATION_TRUNCATE_ROWS = f"{TAG_PREFIX}.alerts.data_exploration_truncate_rows"
    ALERT_DATA_EXPLORATION_TRUNCATE_COLUMNS = f"{TAG_PREFIX}.alerts.data_exploration_truncate_columns"
    ALERT_DATASET_EMPTY = f"{TAG_PREFIX}.alerts.dataset_empty"
    ALERT_DATASET_TOO_LARGE = f"{TAG_PREFIX}.alerts.dataset_too_large"
    ALERT_DATASET_TRUNCATED = f"{TAG_PREFIX}.alerts.dataset_truncated"
    ALERT_DUPLICATE_COLUMN_NAMES = f"{TAG_PREFIX}.alerts.duplicate_col_names"
    ALERT_EXECUTION_TIMEOUT = f"{TAG_PREFIX}.alerts.execution_timeout"
    ALERT_EXTRA_TIME_STEPS_IN_TIME_SERIES = f"{TAG_PREFIX}.alerts.extra_time_steps_in_time_series"
    ALERT_EXTREME_CARDINALITY_COLUMNS = f"{TAG_PREFIX}.alerts.extreme_cardinality_cols"
    ALERT_HIGH_CARDINALITY_COLUMNS = f"{TAG_PREFIX}.alerts.high_cardinality_cols"
    ALERT_HIGH_CORRELATION_COLUMNS = f"{TAG_PREFIX}.alerts.high_correlation_cols"
    ALERT_INAPPROPRIATE_METRIC_FOR_IMBALANCE = f"{TAG_PREFIX}.alerts.inappropriate_metric_for_imbalance"
    ALERT_INCOMPATIBLE_ANNOTATION = f"{TAG_PREFIX}.alerts.incompatible_annotation"
    ALERT_INFERRED_POS_LABEL = f"{TAG_PREFIX}.alerts.inferred_pos_label"
    ALERT_LARGE_NULLS_COLUMNS = f"{TAG_PREFIX}.alerts.large_nulls_cols"
    ALERT_LOW_CARDINALITY_TARGET_COLUMN = f"{TAG_PREFIX}.alerts.low_cardinality_target_col"
    ALERT_MISSING_TIME_STEPS_IN_TIME_SERIES = f"{TAG_PREFIX}.alerts.missing_time_steps_in_time_series"
    ALERT_NO_FEATURE_COLUMNS = f"{TAG_PREFIX}.alerts.no_feature_cols"
    ALERT_NO_PERMISSION_TO_CREATE_SCHEMA = f"{TAG_PREFIX}.alerts.no_permission_to_create_schema"
    ALERT_NO_PERMISSION_TO_CREATE_TABLE = f"{TAG_PREFIX}.alerts.no_permission_to_create_table"
    ALERT_NOT_ENOUGH_HISTORICAL_DATA = f"{TAG_PREFIX}.alerts.not_enough_historical_data"
    ALERT_NULLS_IN_TARGET_COLUMN = f"{TAG_PREFIX}.alerts.nulls_in_target_col"
    ALERT_NULLS_IN_TIME_COLUMN = f"{TAG_PREFIX}.alerts.nulls_in_time_col"
    ALERT_SKEWED_COLUMNS = f"{TAG_PREFIX}.alerts.skewed_cols"
    ALERT_SMALL_NULLS_COLUMNS = f"{TAG_PREFIX}.alerts.small_nulls_cols"
    ALERT_STRONG_CATEGORICAL_TYPE_DETECTION = f"{TAG_PREFIX}.alerts.strong_categorical_type_detection"
    ALERT_STRONG_DATETIME_TYPE_DETECTION = f"{TAG_PREFIX}.alerts.strong_datetime_type_detection"
    ALERT_STRONG_NUMERIC_TYPE_DETECTION = f"{TAG_PREFIX}.alerts.strong_numeric_type_detection"
    ALERT_STRONG_TEXT_TYPE_DETECTION = f"{TAG_PREFIX}.alerts.strong_text_type_detection"
    ALERT_TARGET_LABEL_IMBALANCE = f"{TAG_PREFIX}.alerts.target_label_imbalance"
    ALERT_TARGET_LABEL_INSUFFICIENT_DATA = f"{TAG_PREFIX}.alerts.target_label_insufficient_data"
    ALERT_TARGET_LABEL_RATIO = f"{TAG_PREFIX}.alerts.target_label_ratio"
    ALERT_TIME_SERIES_IDENTITIES_TOO_SHORT = f"{TAG_PREFIX}.alerts.time_series_identities_too_short"
    ALERT_TRAINING_EARLY_STOPPED = f"{TAG_PREFIX}.alerts.training_early_stopped"
    ALERT_TRUNCATE_HORIZON = f"{TAG_PREFIX}.alerts.truncate_horizon"
    ALERT_UNABLE_TO_SAMPLE_WITHOUT_SKEW = f"{TAG_PREFIX}.alerts.unable_to_sample_without_skew"
    ALERT_UNIFORM_COLUMNS = f"{TAG_PREFIX}.alerts.uniform_cols"
    ALERT_UNIQUE_COLUMNS = f"{TAG_PREFIX}.alerts.unique_cols"
    ALERT_UNIQUE_STRING_COLUMNS = f"{TAG_PREFIX}.alerts.unique_string_cols"
    ALERT_UNMATCHED_FREQUENCY_IN_TIME_SERIES = f"{TAG_PREFIX}.alerts.unmatched_frequency_in_time_series"
    ALERT_UNSUPPORTED_FEATURE_COLS = f"{TAG_PREFIX}.alerts.unsupported_feature_cols"
    ALERT_UNSUPPORTED_TARGET_TYPE = f"{TAG_PREFIX}.alerts.unsupported_target_type"
    ALERT_UNSUPPORTED_TIME_TYPE = f"{TAG_PREFIX}.alerts.unsupported_time_type"