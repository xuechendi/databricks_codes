import array
import datetime
import math
import os
import shutil
import tempfile
from datetime import timedelta

import pandas as pd
import requests
import yaml
from pyspark.sql import dataframe
from pyspark.sql.functions import lit

from databricks.feature_store.online_store_spec import (
    AmazonRdsMySqlSpec,
    AzureMySqlSpec,
    AzureSqlServerSpec,
    AmazonDynamoDBSpec,
)


def reset_hive_database(spark_session, db_name):
    drop_hive_database(spark_session, db_name)
    spark_session.sql(f"CREATE DATABASE {db_name}")


def drop_hive_database(spark_session, db_name):
    spark_session.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")


def augment_df(df, column_name, column_int_value):
    return df.withColumn(column_name, lit(column_int_value).cast("int"))


# DataFrame rows -> dict (key is PK)
def df_to_dict(the_df):
    return {r.get("id"): r for r in [r.asDict() for r in the_df.collect()]}


# Perform "upsert"-like combine operation on input dataframes in order
def combine_df_in_order(spark_session, dfs):
    head, *tail = dfs
    expected = df_to_dict(head)
    for tail_df in tail:
        expected.update(df_to_dict(tail_df))
    return spark_session.createDataFrame(expected.values(), head.schema).collect()


# TODO: Add unit tests for assert_pyspark_df_equal method
# check if two pyspark dataframes are equal
def assert_pyspark_df_equal(
    df1: dataframe,
    df2: dataframe,
    check_column_order: bool = True,
    check_row_order: bool = True,
    atol: float = 1e-5,
    time_tol: datetime.timedelta = datetime.timedelta(seconds=1),
    sort_keys: list = [],
):
    # check schema
    # pyspark_df.dtypes is a list of tuple with column name and its type
    if check_column_order:
        assert df1.dtypes == df2.dtypes
    else:
        # ignore column order, sort by column names
        assert sorted(df1.dtypes, key=lambda x: x[0]) == sorted(
            df2.dtypes, key=lambda x: x[0]
        )

    if check_row_order:
        df1_data = df1.collect()
        df2_data = df2.collect()
    else:
        df1_data = (
            df1.sort(sort_keys, ascending=False).collect()
            if sort_keys
            else sorted(df1.collect())
        )
        df2_data = (
            df2.sort(sort_keys, ascending=False).collect()
            if sort_keys
            else sorted(df2.collect())
        )

    def is_equal(c1, c2):
        if type(c1) != type(c2):
            return False
        if type(c1) == float:
            return math.isclose(c1, c2, rel_tol=atol)
        if type(c1) in [datetime.date, datetime.datetime]:
            return abs(c1 - c2) < time_tol
        if type(c1) in [list, array]:
            return all([is_equal(c1[i], c2[i]) for i in range(len(c1))])
        if type(c1) == dict:
            return is_equal(list(c1.keys()), list(c2.keys())) and is_equal(
                list(c1.values()), list(c2.values())
            )
        return c1 == c2

    for r in range(len(df1_data)):
        for c in df1.columns:
            assert is_equal(df1_data[r][c], df2_data[r][c])


# Provider names. If adding or removing a provider, you should also add it to
# PUBLISH_PROVIDERS in FeatureStoreNotebookRunner.scala
PROVIDER_AURORA = "AURORA"
PROVIDER_RDS_MYSQL = "RDS_MYSQL"
PROVIDER_AZURE_MYSQL = "AZURE_MYSQL"
PROVIDER_AZURE_MSSQL = "AZURE_MSSQL"
PROVIDER_DYNAMODB = "DYNAMODB"

# Secret keys
SECRET_KEYS_BY_PROVIDER = {
    PROVIDER_AURORA: {
        "hostname": "AURORA-hostname",
        "port": "AURORA-port",
        "user": "AURORA-user",
        "password": "AURORA-password",
    },
    PROVIDER_RDS_MYSQL: {
        "hostname": "RDS_MYSQL-hostname",
        "port": "RDS_MYSQL-port",
        "user": "RDS_MYSQL-user",
        "password": "RDS_MYSQL-password",
    },
    PROVIDER_AZURE_MYSQL: {
        "hostname": "AZURE_MYSQL-hostname",
        "port": "AZURE_MYSQL-port",
        "user": "AZURE_MYSQL-user",
        "password": "AZURE_MYSQL-password",
    },
    PROVIDER_AZURE_MSSQL: {
        "hostname": "AZURE_MSSQL-hostname",
        "port": "AZURE_MSSQL-port",
        "user": "AZURE_MSSQL-user",
        "password": "AZURE_MSSQL-password",
    },
    PROVIDER_DYNAMODB: {
        "aws_access_key_id": "DYNAMODB-access-key-id",
        "aws_secret_access_key": "DYNAMODB-secret-access-key",
    },
}

# OnlineStoreSpec classes
ONLINE_STORE_SPEC_CLASS_BY_PROVIDER = {
    PROVIDER_AURORA: AmazonRdsMySqlSpec,
    PROVIDER_RDS_MYSQL: AmazonRdsMySqlSpec,
    PROVIDER_AZURE_MYSQL: AzureMySqlSpec,
    PROVIDER_AZURE_MSSQL: AzureSqlServerSpec,
    PROVIDER_DYNAMODB: AmazonDynamoDBSpec,
}

# JDBC Drivers
JDBC_DRIVER_NAME_BY_PROVIDER = {
    PROVIDER_AURORA: "org.mariadb.jdbc.Driver",
    PROVIDER_RDS_MYSQL: "org.mariadb.jdbc.Driver",
    PROVIDER_AZURE_MYSQL: "org.mariadb.jdbc.Driver",
    PROVIDER_AZURE_MSSQL: "com.microsoft.sqlserver.jdbc.SQLServerDriver",
}


def create_online_store_spec(
    provider,
    hostname=None,
    port=None,
    user=None,
    password=None,
    database_name=None,
    table_name=None,
    driver=None,
    read_secret_prefix=None,
    write_secret_prefix=None,
    aws_region="us-west-2",
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    ttl: timedelta = None,
):
    cls = ONLINE_STORE_SPEC_CLASS_BY_PROVIDER[provider]
    if provider == PROVIDER_DYNAMODB:
        return cls(
            region=aws_region,
            access_key_id=aws_access_key_id,
            secret_access_key=aws_secret_access_key,
            session_token=aws_session_token,
            table_name=table_name,
            read_secret_prefix=read_secret_prefix,
            write_secret_prefix=write_secret_prefix,
            ttl=ttl,
        )
    else:
        return cls(
            hostname,
            port,
            user,
            password,
            database_name,
            table_name,
            driver,
            read_secret_prefix,
            write_secret_prefix,
        )


def get_database_user(provider, hostname, user):
    if provider != PROVIDER_AZURE_MYSQL:
        return user
    # https://docs.microsoft.com/en-us/azure/mysql/howto-connection-string
    servername = hostname.split(".")[0]
    return f"{user}@{hostname}"


def get_connection_string(
    provider, database_user, password, hostname, port, database_name=None
):
    if provider != PROVIDER_AZURE_MSSQL:
        protocol = "mysql+pymysql"
    else:
        protocol = "mssql+pyodbc"
        if database_name is None:
            database_name = "master"
    conn = f"{protocol}://{database_user}:{password}@{hostname}:{port}"
    if database_name is not None:
        conn += f"/{database_name}"
    if provider == PROVIDER_AZURE_MSSQL:
        conn += "?driver=ODBC+Driver+17+for+SQL+Server&autocommit=true"
    return conn


def modify_conda_env_yaml(conda_env_yaml_path, pip_dep_prefix, pip_dep_replacement_str):
    """
    Modify conda env yaml in place to replace any pip dependencies that start with pip_dep_prefix
    with pip_dep_replacement_str.

    If there are no pip dependencies that match pip_dep_prefix, then a ValueError will be thrown.

    :param conda_env_yaml_path the fully qualified path to the yaml file that contains the conda environment spec
    :param pip_dep_prefix the prefix of the pip dependency to match against pip dependencies in the conda yml file,
        for example: "databricks-feature-lookup"
    :param pip_dep_replacement_str the new pip dependency to use as a replacement, for example: "https://bucket/file.whl"
    """
    with open(conda_env_yaml_path, "r") as target_artifact_fd:
        conda_yml_content = yaml.safe_load(target_artifact_fd.read())
        error_msg = "No matching pip dependencies found"
        if "dependencies" not in conda_yml_content:
            raise ValueError(
                f"{error_msg}, conda yaml did not contain dependencies section"
            )
        dependencies = conda_yml_content["dependencies"]
        found_pip_deps = False
        for dependency in dependencies:
            # After parsing the conda dependencies there will be two pip dependencies, one that is a
            # string that represents the pip package itself, and another one that is a dictionary that
            # contains the nested pip dependencies.  In this case we want the dictionary with the actual pip
            # dependencies.
            if isinstance(dependency, dict) and "pip" in dependency:
                found_pip_deps = True
                pip_dependencies = dependency["pip"]
                updated_pip_dependencies = []
                found_matching_pip_dep = False
                for pip_dependency in pip_dependencies:
                    if pip_dependency.startswith(pip_dep_prefix):
                        found_matching_pip_dep = True
                        updated_pip_dependencies.append(pip_dep_replacement_str)
                    else:
                        updated_pip_dependencies.append(pip_dependency)
                if not found_matching_pip_dep:
                    raise ValueError(
                        f"{error_msg}, conda yaml did not find a matching pip dependency"
                    )
                dependency["pip"] = updated_pip_dependencies
        if not found_pip_deps:
            raise ValueError(
                f"{error_msg}, conda yaml did not contain a pip dependencies section"
            )
    with open(conda_env_yaml_path, "w") as updated_target_artifact_fd:
        yaml.dump(conda_yml_content, updated_target_artifact_fd)


def modify_pip_requirements(
    pip_requirements_path, pip_dep_prefix, pip_dep_replacement_str
):
    """
    Modify the pip requirements.txt file in place to replace any dependencies that start with
    pip_dep_prefix with pip_dep_replacement_str.

    If a matching pip dependency is not found, this will raise a ValueError.

    This does not handle the following features of the requirements.txt spec:
      - Continuation lines ending in `\`.  These will be treated as separate lines.
      - Including additional files via "-r".  These will be ignored.

    Pip requirements.txt format specification:
    https://pip.pypa.io/en/stable/reference/requirements-file-format/#requirements-file-format)

    :param pip_dep_prefix the prefix of the pip dependency to match against pip dependencies in the conda yml file,
        for example: "databricks-feature-lookup"
    :param pip_dep_replacement_str the new pip dependency to use as a replacement, for example: "https://bucket/file.whl"
    """

    found_pip_dep_prefix = False

    # Copy line-by-line to a temp file, performing desired transformations to each line.
    temp_file_fd, temp_file_path = tempfile.mkstemp()
    with os.fdopen(temp_file_fd, "w") as target_file:
        with open(pip_requirements_path) as source_file:
            for line in source_file:
                if line.strip().startswith(pip_dep_prefix):
                    found_pip_dep_prefix = True
                    target_file.write(pip_dep_replacement_str)
                else:
                    target_file.write(line)

    if not found_pip_dep_prefix:
        os.remove(temp_file_path)
        raise ValueError("No matching pip dependency found")

    # Move temp file with updated contents to overwrite original file
    os.remove(pip_requirements_path)
    shutil.move(temp_file_path, pip_requirements_path)
