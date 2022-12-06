from databricks.feature_store.entities.feature_table import FeatureTable
from databricks.feature_store.utils import request_context
from databricks.feature_store.utils.request_context import RequestContext
from typing import List

import uuid
import inspect

ILLEGAL_TABLE_NAMES = [
    "inv@lidn@me",
    ".table",
    "database.",
    "catalog.database.table.col",
    "d@tabase.table",
    "`database.table`",
    "table.",
    "a.b.c.d",
    "a.b.",
    ".a.b.c",
    "a.b.c.",
    "a!b@c#.d!e@f#",
    "a!b@c#.d!e@f#.g!h@i#",
    "..",
    ".a.",
    "..a",
    "a\.b",
    "",
    None,
]
ILLEGAL_1L_NAMES = ["..", "", None, "db.", ".db", "*&^%"]
VALID_UC_CATALOGS = ["prod_catalog", "dev_catalog", "test", "__catalog123__"]
VALID_DEFAULT_HMS_CATALOGS = ["hive_metastore", "spark_catalog"]

VALID_FEATURE_STORE_1L_TABLE_NAMES = [
    "some_table",
    "___table___",
    "table123",
    "_test_table_123_",
]
VALID_FEATURE_STORE_2L_TABLE_NAMES = [
    "db.table",
    "schema.table",
    "_._",
    "a123_def.123_ddd_fff",
]
VALID_FEATURE_STORE_3L_TABLE_NAMES = [
    "dev.schema.table",
    "prod.database.table",
    "hive_metastore.db.table",
    "spark_catalog.schema.table_1",
    "123_abc.456_def.789_ghi",
    "_._._",
]


def create_test_feature_table(
    name,
    description,
    primary_keys,
    partition_cols,
    features,
    table_id=str(uuid.uuid4()),
    creation_timestamp=0,
    notebook_producers=[],
    job_producers=[],
    timestamp_keys=[],
    path_data_sources=[],
    table_data_sources=[],
    custom_data_sources=[],
):
    return FeatureTable(
        name=name,
        table_id=table_id,
        description=description,
        primary_keys=primary_keys,
        partition_columns=partition_cols,
        timestamp_keys=timestamp_keys,
        features=features,
        creation_timestamp=creation_timestamp,
        online_stores=[],
        notebook_producers=notebook_producers,
        job_producers=job_producers,
        table_data_sources=table_data_sources,
        path_data_sources=path_data_sources,
        custom_data_sources=custom_data_sources,
    )


def assert_request_context(method_calls, expected_feature_store_method_name):
    """
    Assert that every method call in the list of mock.method_call objects that is called
    with a RequestContext parameter is the expected feature_client_method_name.

    :param method_calls: a list of method calls captured by a mock.
    :param expected_feature_store_method_name: the expected feature store method name in the
    the RequestContext parameter of the captured method calls.
    """
    for method_call in method_calls:
        _, positional_args, keyword_args = method_call
        keyword_args_vals = list(keyword_args.values())
        method_params = keyword_args_vals + list(positional_args)
        for method_param in method_params:
            if method_param.__class__ == RequestContext:
                feature_store_method_name = method_param.get_header(
                    request_context.FEATURE_STORE_METHOD_NAME
                )
                assert feature_store_method_name == expected_feature_store_method_name


def get_all_public_methods(obj, white_list: List[str] = []):
    return map(
        lambda f: f[1],
        filter(
            lambda f: callable(f[1])
            and not f[0].startswith("_")
            and not f[0] in white_list,
            inspect.getmembers(obj),
        ),
    )
