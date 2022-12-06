"""
Constants defining the online store property names.

These properties are not stored as properties in the Hive metastore,
so do not need to have prefix 'databricks.feature_store'.
"""
TYPE = "type"
HOSTNAME = "hostname"
PORT = "port"
DATABASE_NAME = "database_name"
TABLE_NAME = "table_name"
FEATURES = "features"
DRIVER_NAME = "driver_name"
READ_SECRET_PREFIX = "read_secret_prefix"
WRITE_SECRET_PREFIX = "write_secret_prefix"
REGION = "region"
TTL = "ttl"
ACCOUNT_URI = "account_uri"
CONTAINER_NAME = "container_name"

# Valid property names that are also valid suffixes for user-provided secret prefixes
USER = "user"
PASSWORD = "password"
ACCESS_KEY_ID = "access-key-id"
SECRET_ACCESS_KEY = "secret-access-key"
SESSION_TOKEN = "session-token"
AUTHORIZATION_KEY = "authorization-key"

# List of explicit credential parameters in any OnlineStoreSpec.
# TODO (ML-23105): Remove explicit parameters for MLR 12.0.
EXPLICIT_CREDENTIAL_PARAMS = [
    USER,
    PASSWORD,
    ACCESS_KEY_ID,
    SECRET_ACCESS_KEY,
    SESSION_TOKEN,
]

# Valid Online Store Types
AWS_MYSQL = "aws.rds.mysql"
AWS_AURORA = "aws.rds.aurora"
AZURE_MYSQL = "azure.db.mysql"
AZURE_SQL_SERVER = "azure.db.mssql"
AWS_DYNAMODB = "aws.dynamodb"
AZURE_COSMOSDB = "azure.cosmosdb"

# Valid Publish Auth Types
SECRETS = "secrets"
ROLE = "role"
