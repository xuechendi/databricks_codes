import base64
import time
import logging
import json

import requests

from databricks.feature_store.version import VERSION
from mlflow.protos import databricks_pb2
from google.protobuf.json_format import MessageToJson, ParseDict
from databricks.feature_store.utils.request_context import RequestContext

_REST_API_PATH_PREFIX = "/api/2.0"
_DEFAULT_HEADERS = {"User-Agent": "feature-store-python-client/%s" % VERSION}

_logger = logging.getLogger(__name__)


def http_request(
    host_creds,
    endpoint,
    retries=3,
    retry_interval=3,
    max_rate_limit_interval=60,
    extra_headers=None,
    **kwargs
):
    """
    Makes an HTTP request with the specified method to the specified hostname/endpoint. Ratelimit
    error code (429) will be retried with an exponential back off (1, 2, 4, ... seconds) for at most
    `max_rate_limit_interval` seconds.  Internal errors (500s) will be retried up to `retries` times
    , waiting `retry_interval` seconds between successive retries. Parses the API response
    (assumed to be JSON) into a Python object and returns it.

    :param extra_headers: a dictionary of extra headers to add to request.
    :param host_creds: Databricks creds containing hostname and optional authentication.
    :return: Parsed API response
    """
    hostname = host_creds.host
    auth_str = None
    if host_creds.username and host_creds.password:
        basic_auth_str = ("%s:%s" % (host_creds.username, host_creds.password)).encode(
            "utf-8"
        )
        auth_str = "Basic " + base64.standard_b64encode(basic_auth_str).decode("utf-8")
    elif host_creds.token:
        auth_str = "Bearer %s" % host_creds.token

    headers = dict(_DEFAULT_HEADERS)

    # Inject any extra headers
    if extra_headers is not None:
        headers.update(extra_headers)

    if auth_str:
        headers["Authorization"] = auth_str

    if host_creds.server_cert_path is None:
        verify = not host_creds.ignore_tls_verification
    else:
        verify = host_creds.server_cert_path

    if host_creds.client_cert_path is not None:
        kwargs["cert"] = host_creds.client_cert_path

    def request_with_ratelimit_retries(max_rate_limit_interval, **kwargs):
        response = requests.request(**kwargs)
        time_left = max_rate_limit_interval
        sleep = 1
        while response.status_code == 429 and time_left > 0:
            _logger.warning(
                "API request to %s returned status code 429 (Rate limit exceeded). "
                "Retrying in %d seconds. "
                "Will continue to retry 429s for up to %d seconds.",
                kwargs.get("url", endpoint),
                sleep,
                time_left,
            )
            time.sleep(sleep)
            time_left -= sleep
            response = requests.request(**kwargs)
            sleep = min(time_left, sleep * 2)  # sleep for 1, 2, 4, ... seconds;
        return response

    cleaned_hostname = hostname[:-1] if hostname.endswith("/") else hostname
    url = "%s%s" % (cleaned_hostname, endpoint)
    for i in range(retries):
        try:
            response = request_with_ratelimit_retries(
                max_rate_limit_interval,
                url=url,
                headers=headers,
                verify=verify,
                **kwargs
            )
            if response.status_code < 500:
                return response
            else:
                _logger.error(
                    "API request to %s failed with code %s, retrying up to %s more times. "
                    "API response body: %s",
                    url,
                    response.status_code,
                    retries - i - 1,
                    response.text,
                )
        # All exceptions that Requests explicitly raises inherit from requests.exceptions.RequestException.
        # See https://docs.python-requests.org/en/latest/user/quickstart/#errors-and-exceptions
        # for more details.
        except requests.exceptions.RequestException as e:
            _logger.error(
                "API request encountered unexpected error: %s. "
                "Requested service might be temporarily unavailable, "
                "retrying up to %s more times.",
                str(e),
                retries - i - 1,
            )
        time.sleep(retry_interval)
    raise Exception("API request to %s failed after %s tries" % (url, retries))


def _can_parse_as_json(string):
    try:
        json.loads(string)
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def verify_rest_response(response, endpoint):
    """Verify the return code and format, raise exception if the request was not successful."""
    if response.status_code != 200:
        if _can_parse_as_json(response.text):
            # ToDo(ML-20622): return cleaner error to client, eg: mlflow.exceptions.RestException
            raise Exception(json.loads(response.text))
        else:
            base_msg = (
                "API request to endpoint %s failed with error code "
                "%s != 200"
                % (
                    endpoint,
                    response.status_code,
                )
            )
            raise Exception("%s. Response body: '%s'" % (base_msg, response.text))

    # Skip validation for endpoints (e.g. DBFS file-download API) which may return a non-JSON
    # response
    if endpoint.startswith(_REST_API_PATH_PREFIX) and not _can_parse_as_json(
        response.text
    ):
        base_msg = (
            "API request to endpoint was successful but the response body was not "
            "in a valid JSON format"
        )
        raise Exception("%s. Response body: '%s'" % (base_msg, response.text))

    return response


def get_error_code(e: Exception):
    if hasattr(e, "args") and len(e.args) > 0 and "error_code" in e.args[0]:
        return e.args[0]["error_code"]
    return None


def get_path(path_prefix, endpoint_path):
    return "{}{}".format(path_prefix, endpoint_path)


def extract_api_info_for_service(service, path_prefix):
    """Return a dictionary mapping each API method to a tuple (path, HTTP method)"""
    service_methods = service.DESCRIPTOR.methods
    res = {}
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        endpoint = endpoints[0]
        endpoint_path = get_path(path_prefix, endpoint.path)
        res[service().GetRequestClass(service_method)] = (
            endpoint_path,
            endpoint.method,
        )
    return res


def json_to_proto(js_dict, message):
    """Parses a JSON dictionary into a message proto, ignoring unknown fields in the JSON."""
    ParseDict(js_dict=js_dict, message=message, ignore_unknown_fields=True)


def proto_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""
    return MessageToJson(message, preserving_proto_field_name=True)


def call_endpoint(host_creds, endpoint, method, json_body, response_proto, req_context):
    # Convert json string to json dictionary, to pass to requests
    if json_body:
        json_body = json.loads(json_body)
    if method == "GET":
        response = http_request(
            host_creds=host_creds,
            endpoint=endpoint,
            method=method,
            params=json_body,
            extra_headers=req_context.get_headers(),
        )
    else:
        response = http_request(
            host_creds=host_creds,
            endpoint=endpoint,
            method=method,
            json=json_body,
            extra_headers=req_context.get_headers(),
        )
    response = verify_rest_response(response, endpoint)
    js_dict = json.loads(response.text)
    json_to_proto(js_dict=js_dict, message=response_proto)
    return response_proto
