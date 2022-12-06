from databricks.automl.internal.clients.client import Client


class JobsClient(Client):
    """
    Jobs client
    """
    _EXPORT_API = "/runs/export"

    def __init__(self, api_url: str, api_token: str):
        """
        :param api_url: The url that can be used to talk to the workspace
        :param api_token: Required auth token
        """
        super().__init__(api_url, api_token, "jobs")

    def export(self, run_id: str) -> str:
        """
        Export the notebook and get the notebook content as HTML
        """
        data = {"run_id": run_id}

        with self.get_request_session() as s:
            resp = s.get(self._base_url + self._EXPORT_API, json=data, auth=self.get_auth())
        if resp.status_code != 200:
            raise Exception(f"Unable to export notebook for job with run_id {run_id}: {resp.text}")

        resp_json = resp.json()
        views = resp_json["views"]
        if len(views) != 1:
            raise Exception(f"Invalid export: expected 1 view but got {len(views)}: {views}")

        view_type = views[0]["type"]
        if view_type != "NOTEBOOK":
            raise Exception(f"Invalid export: expected type to be NOTEBOOK but got {view_type}")

        return views[0]["content"]
