""" Defines the SqlEngine class, which is used to define a common set of
metadata actions that can be performed on SQL databases which inherit
from this class. These metadata actions differ from traditional writes
on the database, which should be done via the Spark SQL JDBC APIs.
"""

import abc

from contextlib import contextmanager
from pyspark.sql.types import StringType


def format_sql_statement(statement):
    stripped = statement.strip()
    if stripped.endswith(";"):
        return stripped
    return stripped + ";"


def get_string_primary_keys(schema, primary_keys):
    return [
        field.name
        for field in schema.fields
        if field.name in primary_keys and isinstance(field.dataType, StringType)
    ]


def set_primary_keys_not_nullable(schema, primary_keys):
    for field in schema.fields:
        if field.name in primary_keys:
            field.nullable = False


class PublishSqlEngine(abc.ABC):

    INFORMATION_SCHEMA = "INFORMATION_SCHEMA"
    TABLES = "TABLES"
    TABLE_CATALOG = "TABLE_CATALOG"
    TABLE_SCHEMA = "TABLE_SCHEMA"
    TABLE_NAME = "TABLE_NAME"
    COLUMNS = "COLUMNS"
    COLUMN_NAME = "COLUMN_NAME"
    DATA_TYPE = "DATA_TYPE"
    NUMERIC_PRECISION = "NUMERIC_PRECISION"
    NUMERIC_SCALE = "NUMERIC_SCALE"

    @abc.abstractmethod
    def __init__(self, online_store, spark_session):
        self.online_store = online_store
        self.spark_session = spark_session

        # This member variable should not be accessed directly. Instead, use
        # self._sql_connection.
        self._cached_sql_connection = None

    def _get_connection(self):
        jvm = self.spark_session.sparkContext._jvm
        if not jvm:
            from pyspark.java_gateway import launch_gateway

            jvm = launch_gateway().jvm
        return jvm.java.sql.DriverManager.getConnection(
            self.jdbc_url,
            self.online_store._lookup_jdbc_auth_user_with_write_permissions(),
            self.online_store._lookup_password_with_write_permissions(),
        )

    @property
    def _sql_connection(self):
        """
        A connection to the SQL database that uses a Java driver.
        We may later update this to use a Python database connector.
        """
        if self._cached_sql_connection:
            return self._cached_sql_connection
        self._cached_sql_connection = self._get_connection()
        return self._cached_sql_connection

    def close(self):
        if self._cached_sql_connection:
            self._sql_connection.close()
            self._cached_sql_connection = None

    def get_online_tables(self, table_names=[]):
        raise NotImplementedError

    def get_column_types(self, table_name):
        raise NotImplementedError

    def add_columns(self, table_name, columns):
        raise NotImplementedError

    def create_table_like(self, table_name, like_table_name, primary_keys):
        raise NotImplementedError

    def rename_table(self, from_name, to_name):
        raise NotImplementedError

    def merge_table_into(self, dst_table_name, src_table_name, columns, primary_keys):
        raise NotImplementedError

    def create_empty_table(
        self, table_name, schema, jdbc_url, connection_properties, primary_keys
    ):
        raise NotImplementedError

    def drop_table(self, table_name):
        self._run_sql_update(f"DROP TABLE IF EXISTS {self._sql_safe_name(table_name)}")

    def add_primary_keys(self, table_name, primary_keys):
        self._run_sql_update(
            f'ALTER TABLE {self._sql_safe_name(table_name)} ADD PRIMARY KEY({", ".join(f"{self._sql_safe_name(pk)}" for pk in primary_keys)})',
        )

    @classmethod
    def _sql_safe_name(cls, name):
        raise NotImplementedError

    def _make_empty_df(self, schema):
        return self.spark_session.createDataFrame([], schema)

    @property
    def jdbc_url(self):
        raise NotImplementedError

    @property
    def jdbc_properties(self):
        raise NotImplementedError

    @contextmanager
    def in_transaction(self):
        self._sql_connection.setAutoCommit(False)  # Start transaction
        try:
            yield
            self._sql_connection.commit()  # End transaction
        except:
            self._sql_connection.rollback()
            raise
        finally:
            self._sql_connection.setAutoCommit(True)

    def _run_sql_update(self, statement):
        stmt = self._sql_connection.createStatement()
        stmt.executeUpdate(format_sql_statement(statement))

    def _run_sql_query(self, query):
        stmt = self._sql_connection.createStatement()
        return stmt.executeQuery(format_sql_statement(query))
