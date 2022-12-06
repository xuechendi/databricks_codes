from typing import Union

import pandas as pd
import pyspark.pandas as ps
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession

from databricks.automl.shared.const import GLOBAL_TEMP_DATABASE
from databricks.automl.shared.errors import UnsupportedDataError


def use_automl_client() -> bool:
    """
    A temporary feature flag.
    When true, this package should use the automl client to start an experiment.
    When false, this package should skip the client and starts an experiment directly using internal or legacy API.

    When we can confidently enable the client by default, we can delete all the old code by checking for all
    calls to this function.
    """
    spark = SparkSession.builder.getOrCreate()
    return spark.conf.get("spark.databricks.automl.useClient", "") != "false"


def is_automl_service_enabled() -> bool:
    """
    This conf is set in
    universe/manager/src/main/scala/com/databricks/backend/manager/SparkEnvironment.scala:getAutomlServiceEnabledConf

    This conf will be true on MT/GFM, where the automl service runs the driver notebook,
    and will be false on ST/PVC, where the webapp frontend runs the driver notebook.
    """
    spark = SparkSession.builder.getOrCreate()
    return spark.conf.get("spark.databricks.automl.serviceEnabled", "") == "true"


def convert_to_spark_dataframe(dataset: Union[DataFrame, pd.DataFrame, ps.DataFrame]) -> DataFrame:
    """
    Validates the input dataset and returns a converted
    spark dataframe if the input is a pandas dataframe
    :param dataset: Either a Spark or a pandas DataFrame
    :return: Spark DataFrame
    """
    spark = SparkSession.builder.getOrCreate()
    if isinstance(dataset, DataFrame):
        return dataset
    if isinstance(dataset, pd.DataFrame):
        return spark.createDataFrame(dataset)
    if isinstance(dataset, ps.DataFrame):
        return dataset.to_spark()
    raise UnsupportedDataError(
        f"input dataset is not a pyspark DataFrame, pyspark.pandas DataFrame or a pandas DataFrame: {type(dataset)}"
    )


def drop_if_global_temp_view(dataset: str):
    """
    Deletes the global temp view, if one exists.
    spark.catalog.dropGlobalTempView can be called multiple times and just no-ops if the table name doesn't exist
    """
    spark = SparkSession.builder.getOrCreate()
    if dataset.startswith(GLOBAL_TEMP_DATABASE):
        temp_view_name = dataset[len(f"{GLOBAL_TEMP_DATABASE}."):]
        spark.catalog.dropGlobalTempView(temp_view_name)
