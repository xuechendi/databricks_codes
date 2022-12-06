import requests.auth
from databricks.automl.client.protos.common_pb2 import Experiment
from databricks.automl.client.protos.service_pb2 import CreateExperiment, GetExperiment
from google.protobuf.json_format import MessageToDict, ParseDict
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from databricks.automl.shared.errors import AutomlServiceError, InvalidArgumentError


class AutomlServiceClient:
    """
    Client to talk to the automl service.
    """

    def __init__(self, api_url: str, api_token: str):
        """
        :param api_url:   The base api url fetched from dbutils
        :param api_token: The auth token fetched from dbutils
        """
        self._api_url = f"{api_url}/api/2.0/automl"
        self._api_token = api_token

    class BearerAuth(requests.auth.AuthBase):
        """
        Bearer Auth class for providing authentication
        """

        def __init__(self, token: str):
            self._token = token

        def __call__(self, r):
            r.headers["Authorization"] = f"Bearer {self._token}"
            return r

    def _get_create_experiment_url(self) -> str:
        return f"{self._api_url}/experiments"

    def _get_get_experiment_url(self, experiment_id: str) -> str:
        return f"{self._api_url}/experiments/{experiment_id}"

    def _get_cancel_experiment_url(self, experiment_id: str) -> str:
        return f"{self._api_url}/experiments/{experiment_id}/cancel"

    def _get_auth(self):
        return self.BearerAuth(self._api_token)

    def _get_headers(self):
        return {'X-Source': 'python'}

    @staticmethod
    def _get_request_session() -> requests.Session:
        retry = Retry(
            total=6,
            # sleep between retries will be:
            #   {backoff factor} * (2 ** ({number of total retries} - 1))
            # so this will sleep for [5, 10, 20, 40, 80, 160..] seconds
            # which is a total of 315 seconds of wait time with 6 retries
            # which is a total of > 5 minutes of wait time for workspace / jobs
            # to restart or recover from downtime
            backoff_factor=5,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"])

        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def create_experiment(self, create_experiment_proto: CreateExperiment) -> str:
        """
        Create an AutoML experiment

        :param create_experiment_proto:   Proto with params specific to the problem type
        :return: Experiment Id of the created experiment
        """

        # preserving_proto_field_name=True ensures that keys
        # for the resulting json are snake_case instead of camelCase
        data = MessageToDict(create_experiment_proto, preserving_proto_field_name=True)
        with self._get_request_session() as s:
            resp = s.post(
                self._get_create_experiment_url(),
                json=data,
                auth=self._get_auth(),
                headers=self._get_headers())

        # special handing of INVALID_PARAMETER_VALUE to allow users to
        # easily identify user error vs. service error
        if resp.status_code == 400:
            raise InvalidArgumentError(f"Invalid argument passed to AutoML: Error: {resp.content}")
        if resp.status_code != 200:
            raise AutomlServiceError("Failed to create Automl experiment. "
                                     f"Status Code: {resp.status_code} Error: {resp.content}")
        resp_proto = ParseDict(resp.json(), CreateExperiment.Response(), ignore_unknown_fields=True)
        return resp_proto.experiment_id

    def get_experiment(self, experiment_id: str) -> Experiment:
        with self._get_request_session() as s:
            resp = s.get(
                self._get_get_experiment_url(experiment_id),
                auth=self._get_auth(),
                headers=self._get_headers())

        if resp.status_code != 200:
            raise AutomlServiceError(f"Failed to fetch experiment with id: {experiment_id} "
                                     f"Status Code: {resp.status_code} Error: {resp.content}")
        resp_proto = ParseDict(resp.json(), GetExperiment.Response(), ignore_unknown_fields=True)
        return resp_proto.experiment

    def cancel_experiment(self, experiment_id: str) -> None:
        with self._get_request_session() as s:
            resp = s.post(
                self._get_cancel_experiment_url(experiment_id),
                auth=self._get_auth(),
                headers=self._get_headers())

        if resp.status_code != 200:
            raise AutomlServiceError(f"Failed to cancel experiment with id: {experiment_id} "
                                     f"Status Code: {resp.status_code} Error: {resp.content}")
