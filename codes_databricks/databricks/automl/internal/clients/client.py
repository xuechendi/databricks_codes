import requests.auth
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class Client:
    """
    Abstract client to talk to Databricks
    """

    def __init__(self, api_url: str, api_token: str, endpoint: str):
        """
        :param api_url: The url that can be used to talk to the workspace
        :param api_token: Required auth token
        :param endpoint: Endpoint to construct base url
        """
        self._api_url = api_url
        self._api_token = api_token
        self._base_url = f"{self._api_url}/api/2.0/{endpoint}"

    def get_auth(self):
        """
        Get a BearerAuth for requests
        """
        # TODO(ML-12218): api_token expires in 24hrs or less; switch to credential provider
        return self.BearerAuth(self._api_token)

    @staticmethod
    def get_request_session() -> requests.Session:
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

    class BearerAuth(requests.auth.AuthBase):
        def __init__(self, token: str):
            self._token = token

        def __call__(self, r):
            r.headers["authorization"] = f"Bearer {self._token}"
            return r
