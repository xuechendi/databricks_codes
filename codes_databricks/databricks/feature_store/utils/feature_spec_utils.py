from databricks.feature_store.entities.feature_spec import FeatureSpec
from databricks.feature_store.entities.feature_column_info import FeatureColumnInfo
from databricks.feature_store.entities.feature_table_info import FeatureTableInfo
from databricks.feature_store.entities.source_data_column_info import (
    SourceDataColumnInfo,
)

from typing import List, Union

TEST_WORKSPACE_ID = 123


def get_test_table_info_from_column_info(
    column_infos: List[Union[FeatureColumnInfo, SourceDataColumnInfo]]
):
    table_infos = []
    unique_table_names = set(
        [
            column_info.table_name
            for column_info in column_infos
            if isinstance(column_info, FeatureColumnInfo)
        ]
    )
    for table_name in unique_table_names:
        table_id = table_name + "123456"
        table_infos.append(FeatureTableInfo(table_name=table_name, table_id=table_id))
    return table_infos


def create_test_feature_spec(
    column_infos: List[Union[FeatureColumnInfo, SourceDataColumnInfo]],
    table_infos: List[FeatureTableInfo] = None,
    workspace_id: int = TEST_WORKSPACE_ID,
):
    if table_infos is None:
        table_infos = get_test_table_info_from_column_info(column_infos)
    return FeatureSpec(
        column_infos=column_infos,
        table_infos=table_infos,
        workspace_id=workspace_id,
        feature_store_client_version="test0",
    )
