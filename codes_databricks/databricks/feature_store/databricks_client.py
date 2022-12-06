from databricks.feature_store.utils.rest_utils import http_request, verify_rest_response

_GET_JOB_ENDPOINT = "/api/2.0/jobs/get"


class DatabricksClient:
    """
    This client issues requests to services other than Feature Store.

    If the number of APIs accessed grows too large, we should split this up into per-service APIs
    and import the request/repsonse protos for validation.
    """

    def __init__(self, get_host_creds):
        self._get_host_creds = get_host_creds

    def take_notebook_snapshot(self, notebook_path: str):
        """
        Sends a request to WebApp to take a notebook snapshot, and returns the revision timestamp
        of the snapshot.

        This method should be updated to route the call through the Feature Store service, as
        mlflow does.
        """
        response = http_request(
            self._get_host_creds(),
            "/api/2.0/workspace/take-notebook-snapshot",  # Inlined because this is not a public API
            method="POST",
            json={"hidden": True, "path": notebook_path},
        )
        response = verify_rest_response(
            response, "/api/2.0/workspace/take-notebook-snapshot"
        )
        # It would be better to deserialize the response as a proto, but the Feature Store python
        # wheel should not leak other internal WebApp APIs.
        return response.json()["revision"]["revision_timestamp"]
