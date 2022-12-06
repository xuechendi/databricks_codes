import copy

from databricks.feature_store.constants import MAX_PRIMARY_KEY_STRING_LENGTH_CHARS
from databricks.feature_store.publish_engine.publish_sql_engine import (
    PublishSqlEngine,
    get_string_primary_keys,
    set_primary_keys_not_nullable,
)
from databricks.feature_store.utils.spark_utils import (
    get_columns_of_type_short,
    serialize_complex_data_types,
)


class PublishMySqlEngine(PublishSqlEngine):
    CHARSET = "utf8mb4"
    COLLATION = "utf8mb4_bin"
    PRIMARY_KEY_STRING_ERROR = "Data too long for column"
    MAX_VARCHAR_LENGTH_CHARS = 1000

    def __init__(self, online_store, spark_session):
        super().__init__(online_store, spark_session)

    def get_online_tables(self, table_names=[]):
        query = (
            f"SELECT {PublishSqlEngine.TABLE_NAME} FROM {PublishSqlEngine.INFORMATION_SCHEMA}.{PublishSqlEngine.TABLES} "
            f"WHERE {PublishSqlEngine.TABLE_SCHEMA}='{self.online_store.database_name}'"
        )
        if len(table_names) > 0:
            table_names_str = ", ".join(
                [f"'{table_name}'" for table_name in table_names]
            )
            query += f" AND {PublishSqlEngine.TABLE_NAME} IN ({table_names_str})"
        return self._run_sql_query(query)

    def get_column_types(self, table_name):
        query = (
            f"SELECT {PublishSqlEngine.COLUMN_NAME}, {PublishSqlEngine.DATA_TYPE}, {PublishSqlEngine.NUMERIC_PRECISION}, {PublishSqlEngine.NUMERIC_SCALE} "
            f"FROM {PublishSqlEngine.INFORMATION_SCHEMA}.{PublishSqlEngine.COLUMNS} "
            f"WHERE {PublishSqlEngine.TABLE_SCHEMA}='{self.online_store.database_name}' "
            f"AND {PublishSqlEngine.TABLE_NAME}='{table_name}'"
        )
        return self._run_sql_query(query)

    def add_columns(self, table_name, columns):
        if len(columns) == 0:
            return
        new_columns_str = ", ".join(
            [f"{self._sql_safe_name(name)} {type}" for (name, type) in columns]
        )
        self._run_sql_update(
            f"ALTER TABLE {self._sql_safe_name(table_name)} ADD COLUMN ({new_columns_str})"
        )

    def create_table_like(self, table_name, like_table_name, primary_keys):
        self._run_sql_update(
            f"CREATE TABLE IF NOT EXISTS {self._sql_safe_name(table_name)} LIKE {self._sql_safe_name(like_table_name)}",
        )

    def rename_table(self, from_name, to_name):
        self._run_sql_update(
            f"ALTER TABLE {self._sql_safe_name(from_name)} RENAME TO {self._sql_safe_name(to_name)}"
        )

    def merge_table_into(self, dst_table_name, src_table_name, columns, primary_keys):
        columns_str = ", ".join(f"{self._sql_safe_name(c)}" for c in columns)
        update_columns = [col for col in columns if col not in primary_keys]
        update_str = ", ".join(
            [
                f"{self._sql_safe_name(feat)}={self._sql_safe_name(src_table_name)}.{self._sql_safe_name(feat)}"
                for feat in update_columns
            ]
        )

        update_cmd = (
            f"INSERT INTO {self._sql_safe_name(dst_table_name)} ({columns_str}) "
            f"SELECT {columns_str} FROM {self._sql_safe_name(src_table_name)} "
            f"ON DUPLICATE KEY UPDATE {update_str}"
        )
        self._run_sql_update(update_cmd)

    def _update_create_table_column_types_option(
        self, writer, string_pk_cols, short_cols
    ):
        string_pk_col_types = [
            f"{col} VARCHAR({MAX_PRIMARY_KEY_STRING_LENGTH_CHARS})"
            for col in string_pk_cols
        ]
        # Explicitly specify the SHORT type to prevent upcasting to INTEGER by the Spark writer
        # SHORT is specified rather than MySql's SMALLINT as the documentation asks for a Spark Sql type
        # However, SHORT and SMALLINT are equivalent. SHORT will be written as SMALLINT in MySql
        # https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html
        short_col_types = [f"{col} SHORT" for col in short_cols]

        all_col_types = string_pk_col_types + short_col_types
        if len(all_col_types) > 0:
            writer = writer.option("createTableColumnTypes", ", ".join(all_col_types))
        return writer

    def create_empty_table(
        self, table_name, schema, jdbc_url, connection_properties, primary_keys
    ):
        schema = copy.deepcopy(schema)
        set_primary_keys_not_nullable(schema, primary_keys)
        empty_df = self._make_empty_df(schema)

        string_pk_cols = get_string_primary_keys(schema, primary_keys)
        short_cols = get_columns_of_type_short(schema)

        serialized_df = serialize_complex_data_types(empty_df)
        writer = self._update_create_table_column_types_option(
            serialized_df.write, string_pk_cols, short_cols
        )
        writer.option(
            "createTableOptions",
            f"DEFAULT CHARSET={self.CHARSET} COLLATE={self.COLLATION}",
        ).jdbc(
            url=jdbc_url,
            table=table_name,
            mode="overwrite",
            properties=connection_properties,
        )
        self.add_primary_keys(table_name, primary_keys)

    @classmethod
    def _sql_safe_name(cls, name):
        # MySQL requires `xxx` format to safely handle identifiers that contain special characters or are reserved words.
        return f"`{name}`"

    @property
    def jdbc_url(self):
        return f"jdbc:mysql://{self.online_store.hostname}:{self.online_store.port}/{self.online_store.database_name}"

    @property
    def jdbc_properties(self):
        properties = {
            "user": self.online_store._lookup_jdbc_auth_user_with_write_permissions(),
            "password": self.online_store._lookup_password_with_write_permissions(),
        }
        if self.online_store.driver:
            properties["driver"] = self.online_store.driver
        return properties
