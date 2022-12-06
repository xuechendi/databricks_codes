import copy

from databricks.feature_store.constants import MAX_PRIMARY_KEY_STRING_LENGTH_BYTES
from databricks.feature_store.publish_engine.publish_sql_engine import (
    PublishSqlEngine,
    get_string_primary_keys,
    set_primary_keys_not_nullable,
)
from databricks.feature_store.utils.spark_utils import (
    get_columns_of_type_short,
    serialize_complex_data_types,
)


class PublishSqlServerEngine(PublishSqlEngine):
    """
    SqlServerEngine supports SQL Server 2019 and newer due to its reliance on UTF-8 collations.
    (https://docs.microsoft.com/en-us/sql/relational-databases/collations/collation-and-unicode-support?view=sql-server-2017#utf8)
    """

    DEFAULT_DRIVER = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    COLLATION = "Latin1_General_100_BIN2_UTF8"
    MAX_NUM_STRING_PK_COLUMNS = 2
    PRIMARY_KEY_STRING_ERROR = "String or binary data would be truncated"

    def __init__(self, online_store, spark_session):
        super().__init__(online_store, spark_session)

    def get_online_tables(self, table_names=[]):
        query = (
            f"SELECT {PublishSqlEngine.TABLE_NAME} "
            f"FROM {PublishSqlEngine.INFORMATION_SCHEMA}.{PublishSqlEngine.TABLES} "
            f"WHERE {PublishSqlEngine.TABLE_CATALOG}='{self.online_store.database_name}'"
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
            f"WHERE {PublishSqlEngine.TABLE_CATALOG}='{self.online_store.database_name}' AND {PublishSqlEngine.TABLE_NAME}='{table_name}'"
        )
        return self._run_sql_query(query)

    def add_columns(self, table_name, columns):
        if len(columns) == 0:
            return
        new_columns_str = ", ".join(
            [f"{self._sql_safe_name(name)} {type}" for (name, type) in columns]
        )
        self._run_sql_update(
            f"ALTER TABLE {self._sql_safe_name(table_name)} ADD {new_columns_str}"
        )

    def create_table_like(self, table_name, like_table_name, primary_keys):
        with self.in_transaction():
            primary_keys_str = ", ".join(
                f"{self._sql_safe_name(pk)}" for pk in primary_keys
            )
            update_cmd = (
                "IF NOT EXISTS "
                f"(SELECT * FROM {PublishSqlEngine.INFORMATION_SCHEMA}.{PublishSqlEngine.TABLES} WHERE {PublishSqlEngine.TABLE_NAME} = '{table_name}') "
                "BEGIN "
                f"SELECT * INTO {self._sql_safe_name(table_name)} FROM {self._sql_safe_name(like_table_name)} WHERE 0 = 1; "
                f"ALTER TABLE {self._sql_safe_name(table_name)} ADD PRIMARY KEY({primary_keys_str}); "
                "END"
            )
            return self._run_sql_update(update_cmd)

    def rename_table(self, from_name, to_name):
        self._run_sql_update(
            f"EXEC sp_rename {self._sql_safe_name(from_name)}, {self._sql_safe_name(to_name)}"
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
        primary_keys_str = " AND ".join(
            [
                f"{self._sql_safe_name(dst_table_name)}.{self._sql_safe_name(key)}={self._sql_safe_name(src_table_name)}.{self._sql_safe_name(key)}"
                for key in primary_keys
            ]
        )
        values_str = ", ".join(
            [
                f"{self._sql_safe_name(src_table_name)}.{self._sql_safe_name(col)}"
                for col in columns
            ]
        )

        update_cmd = (
            f"MERGE {self._sql_safe_name(dst_table_name)} "
            f"USING {self._sql_safe_name(src_table_name)} ON {primary_keys_str} "
            "WHEN MATCHED THEN "
            f"UPDATE SET {update_str} "
            "WHEN NOT MATCHED BY TARGET THEN "
            f"INSERT ({columns_str}) VALUES ({values_str})"
        )
        self._run_sql_update(update_cmd)

    def _alter_string_pk_collation(self, table_name, primary_key):
        self._run_sql_update(
            f"ALTER TABLE {self._sql_safe_name(table_name)} "
            f"ALTER COLUMN {self._sql_safe_name(primary_key)} VARCHAR({MAX_PRIMARY_KEY_STRING_LENGTH_BYTES}) "
            f"COLLATE {self.COLLATION} "
            f"NOT NULL"
        )

    def _update_create_table_column_types_option(self, writer, short_cols):
        # Explicitly specify the SHORT type to prevent upcasting to INTEGER by the Spark writer
        # SHORT is specified rather than Sql Server's SMALLINT as the documentation asks for a Spark Sql type
        # However, SHORT and SMALLINT are equivalent. SHORT will be written as SMALLINT in Sql Server
        # https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html
        if len(short_cols) > 0:
            short_col_types = [f"{col} SHORT" for col in short_cols]
            writer = writer.option(
                "createTableColumnTypes",
                ", ".join(short_col_types),
            )
        return writer

    def create_empty_table(
        self, table_name, schema, jdbc_url, connection_properties, primary_keys
    ):
        # MSSQL requires columns be non-nullable before setting them as
        # primary keys. We set the primary key fields in the schema
        # to be non-nullable so they are defined as not-nullable
        # in the online store schema.
        schema = copy.deepcopy(schema)
        set_primary_keys_not_nullable(schema, primary_keys)
        empty_df = self._make_empty_df(schema)

        string_pk_cols = get_string_primary_keys(schema, primary_keys)
        short_cols = get_columns_of_type_short(schema)

        # SQL Server 2019 does not error out if the primary key maximum length (900 bytes) is exceeded.
        # Therefore, we explicitly guard it from feature tables that have more than two
        # 400-byte string columns in the primary key. Exceeding the maximum length is still
        # possible using additional column types but is unlikely.
        if len(string_pk_cols) > self.MAX_NUM_STRING_PK_COLUMNS:
            raise ValueError(
                f"Feature tables with more than {self.MAX_NUM_STRING_PK_COLUMNS} string columns "
                f"in their primary key cannot be published to SQL Server; "
                f"found {len(string_pk_cols)} string columns in the primary key."
            )

        serialized_df = serialize_complex_data_types(empty_df)
        writer = self._update_create_table_column_types_option(
            serialized_df.write, short_cols
        )
        writer.jdbc(
            url=jdbc_url,
            table=table_name,
            mode="overwrite",
            properties=connection_properties,
        )
        for string_pk in string_pk_cols:
            self._alter_string_pk_collation(table_name, string_pk)

        self.add_primary_keys(table_name, primary_keys)

    @classmethod
    def _sql_safe_name(cls, name):
        # MSSQL requires [xxx] format to safely handle identifiers that contain special characters or are reserved words.
        return f"[{name}]"

    @property
    def jdbc_url(self):
        return f"jdbc:sqlserver://{self.online_store.hostname}:{self.online_store.port};database={self.online_store.database_name}"

    @property
    def jdbc_properties(self):
        return {
            "user": self.online_store._lookup_jdbc_auth_user_with_write_permissions(),
            "password": self.online_store._lookup_password_with_write_permissions(),
            "driver": self.online_store.driver or self.DEFAULT_DRIVER,
        }
