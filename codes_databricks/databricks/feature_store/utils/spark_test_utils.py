from unittest.mock import MagicMock

from pyspark.sql import DataFrame


def mock_pyspark_dataframe(**kwargs):
    df = MagicMock(spec=DataFrame)
    df.configure_mock(**kwargs)
    return df
