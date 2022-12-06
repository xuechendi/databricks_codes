from mlflow.utils import databricks_utils
import logging
import pkg_resources
import yaml
from typing import List

_logger = logging.getLogger(__name__)

# Load in the list of valid default header keys from the headers.yaml bundled resource
constants_yml = pkg_resources.resource_string(__name__, "headers.yaml")
if constants_yml is None or len(constants_yml) == 0:
    raise Exception(
        "Missing headers.yaml package resource.  This indicates a packaging error."
    )
constants = yaml.safe_load(constants_yml)
valid_header_keys = {}
for key in constants.keys():
    valid_header_keys[key] = set(constants.get(key, []))

# Default header key for the feature store client API method that originated this request context
FEATURE_STORE_METHOD_NAME = "feature-store-method-name"

# Other default header keys
CLUSTER_ID = "cluster_id"
NOTEBOOK_ID = "notebook_id"
JOB_ID = "job_id"
JOB_RUN_ID = "job_run_id"
JOB_TYPE = "job_type"

# custom header keys
PUBLISH_AUTH_TYPE = "publish_auth_type"

DEFAULT_HEADERS = "default_headers"
# Valid header keys and header values for FEATURE_STORE_METHOD_NAME header key.
GET_TABLE = "get_table"
GET_FEATURE_TABLE = "get_feature_table"
CREATE_TABLE = "create_table"
CREATE_FEATURE_TABLE = "create_feature_table"
REGISTER_TABLE = "register_table"
PUBLISH_TABLE = "publish_table"
WRITE_TABLE = "write_table"
READ_TABLE = "read_table"
DROP_TABLE = "drop_table"
LOG_MODEL = "log_model"
SCORE_BATCH = "score_batch"
CREATE_TRAINING_SET = "create_training_set"
GET_MODEL_SERVING_METADATA = "get_model_serving_metadata"
SET_FEATURE_TABLE_TAG = "set_feature_table_tag"
DELETE_FEATURE_TABLE_TAG = "delete_feature_table_tag"
ADD_DATA_SOURCES = "add_data_sources"
DELETE_DATA_SOURCES = "delete_data_sources"
TEST_ONLY_METHOD = "test_only_method"


class RequestContext:

    """
    An object for instrumenting the feature store client usage patterns.  Client methods in the public
    API should create a RequestContext and pass it down the callstack to the catalog client, which will
    add all relevant context to the outgoing requests as HTTP headers.  The catalog service will read the
    headers and record in usage logs.
    """

    # The list of valid header values for the FEATURE_STORE_METHOD_NAME header key
    valid_feature_store_method_names = [
        GET_TABLE,
        GET_FEATURE_TABLE,
        CREATE_TABLE,
        CREATE_FEATURE_TABLE,
        PUBLISH_TABLE,
        WRITE_TABLE,
        READ_TABLE,
        REGISTER_TABLE,
        DROP_TABLE,
        LOG_MODEL,
        SCORE_BATCH,
        CREATE_TRAINING_SET,
        GET_MODEL_SERVING_METADATA,
        SET_FEATURE_TABLE_TAG,
        DELETE_FEATURE_TABLE_TAG,
        ADD_DATA_SOURCES,
        DELETE_DATA_SOURCES,
        TEST_ONLY_METHOD,
    ]

    def __init__(self, feature_store_method_name: str, custom_headers: dict = {}):
        """
        Initializer

        :param feature_store_method_name: The feature store method creating this request context.
        """
        if (
            feature_store_method_name
            not in RequestContext.valid_feature_store_method_names
        ):
            raise ValueError(
                f"Invalid feature store method name given: {feature_store_method_name}"
            )

        # Create default headers
        default_headers = self._create_default_headers(feature_store_method_name)

        # Ensure that no header keys outside of those declared in headers.yaml have snuck into the codebase
        self._validate_headers(
            feature_store_method_name,
            default_headers,
            custom_headers,
            valid_header_keys,
        )

        # Save default in internal headers map
        self._headers = default_headers
        # Save custom headers
        self._headers.update(custom_headers)

    def __eq__(self, other):
        """
        Override equality testing to compare the internal state rather than comparing by reference.
        Curently only needed for testing purposes.  If additional state variables are added to this
        object this method will need to be updated accordingly.
        """
        if not isinstance(other, RequestContext):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def _create_default_headers(self, feature_store_method_name):
        """
        Create the default headers that will be sent with every RPC request from the client.
        """
        default_headers = {FEATURE_STORE_METHOD_NAME: feature_store_method_name}
        try:
            if databricks_utils.is_in_cluster():
                default_headers[CLUSTER_ID] = databricks_utils.get_cluster_id()
            if databricks_utils.is_in_databricks_notebook():
                default_headers[NOTEBOOK_ID] = databricks_utils.get_notebook_id()
            if databricks_utils.is_in_databricks_job():
                default_headers[JOB_ID] = databricks_utils.get_job_id()
                default_headers[JOB_RUN_ID] = databricks_utils.get_job_run_id()
                default_headers[JOB_TYPE] = databricks_utils.get_job_type()
        except Exception as e:
            _logger.warning(
                "Exeption while adding standard headers, some headers will not be added.",
                exc_info=e,
            )
        return default_headers

    def _validate_headers(
        self,
        feature_store_method_name,
        default_headers,
        custom_headers,
        valid_header_keys,
    ):
        """
        Ensure that all headers are in the list of valid headers expected by the catalog service.
        This prevents any headers being added to the client while forgetting to add to the
        catalog service, since both share the headers.yaml file.
        """
        unknown_header_keys = []
        for key in default_headers.keys():
            if key not in valid_header_keys[DEFAULT_HEADERS]:
                unknown_header_keys.append(key)
        for key in custom_headers.keys():
            if (feature_store_method_name not in valid_header_keys) or (
                key not in valid_header_keys[feature_store_method_name]
            ):
                unknown_header_keys.append(key)
        if len(unknown_header_keys) > 0:
            raise Exception(
                f'Unknown header key{"s" if len(unknown_header_keys) > 1 else ""}: '
                f'{", ".join(unknown_header_keys)}. Please add to headers.yaml'
            )

    def get_header(self, header_name: str):
        """
        Get the specified header, or return None if there is no corresponding header.
        """
        return self._headers.get(header_name)

    def get_headers(self) -> map:
        """
        Get the stored headers.
        """
        return dict.copy(self._headers)
