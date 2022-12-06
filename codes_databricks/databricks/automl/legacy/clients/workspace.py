import base64
import re

import nbformat
from databricks.automl.legacy.client import Client
from databricks.automl.legacy.errors import NotebookImportSizeExceededError


class WorkspaceClient(Client):
    """
    Client to talk to the workspace to read / write notebooks
    """
    _MKDIRS_API = "/mkdirs"
    _IMPORT_API = "/import"
    _EXPORT_API = "/export"
    _LIST_API = "/list"
    _GET_STATUS_API = "/get-status"
    _DELETE_API = "/delete"

    def __init__(self, api_url: str, api_token: str):
        """
        :param api_url: The url that can be used to talk to the workspace
        :param api_token: Required auth token
        """
        super().__init__(api_url, api_token, "workspace")

    def mkdirs(self, path: str) -> None:
        data = {"path": path}
        with self.get_request_session() as s:
            resp = s.post(self._base_url + self._MKDIRS_API, json=data, auth=self.get_auth())

        if resp.status_code != 200:
            raise Exception(f"Unable to create parent directory at {path}: {resp.text}")

    def exists(self, path: str) -> bool:
        data = {"path": path}
        with self.get_request_session() as s:
            resp = s.get(self._base_url + self._LIST_API, json=data, auth=self.get_auth())

        # if the path exists, list returns 200 else it returns 404
        return resp.status_code == 200

    def import_nbformat(self, path: str, notebook: nbformat.NotebookNode) -> None:
        content = nbformat.writes(notebook)
        self._import_notebook(path, content, "JUPYTER")

    def get_notebook_url(self, path: str) -> str:
        notebook_id = self.get_notebook_id(path)
        return f"#notebook/{notebook_id}"

    def get_notebook_id(self, path: str) -> int:
        data = {"path": path}

        with self.get_request_session() as s:
            resp = s.get(self._base_url + self._GET_STATUS_API, params=data, auth=self.get_auth())

        if resp.status_code != 200:
            raise Exception(f"Unable to fetch notebook url from path {path}: {resp.text}")

        return resp.json()["object_id"]

    def import_html(self, path: str, html: str) -> None:
        self._import_notebook(path, html, "HTML")

    def _import_notebook(self, path: str, content: str, content_format: str) -> None:
        data = {
            "path": path,
            "format": content_format,
            "content": base64.standard_b64encode(bytes(content, encoding="utf-8")),
            "overwrite": "true",
        }

        with self.get_request_session() as s:
            resp = s.post(self._base_url + self._IMPORT_API, json=data, auth=self.get_auth())

        if resp.status_code == 400 and \
                re.match(".*content size.*exceeded the limit.*", resp.content.decode("utf-8")) is not None:
            raise NotebookImportSizeExceededError

        if resp.status_code != 200:
            raise Exception(
                f"Unable to generate notebook at {path} using format {content_format}: {resp.text}")

    def export_nbformat(self, path: str) -> nbformat.NotebookNode:
        content = self._export_notebook(path, "JUPYTER")
        return nbformat.reads(content, as_version=4)

    def _export_notebook(self, path: str, content_format: str) -> str:
        data = {
            "path": path,
            "format": content_format,
        }

        with self.get_request_session() as s:
            resp = s.get(self._base_url + self._EXPORT_API, json=data, auth=self.get_auth())
        if resp.status_code != 200:
            raise Exception(
                f"Unable to export notebook at {path} using format {content_format}: {resp.text}")

        content_b64 = resp.json()["content"]
        return base64.standard_b64decode(content_b64).decode("utf-8")

    def delete(self, path: str, recursive: bool = False) -> None:
        data = {"path": path, "recursive": recursive}
        with self.get_request_session() as s:
            resp = s.post(self._base_url + self._DELETE_API, json=data, auth=self.get_auth())
        if resp.status_code != 200:
            raise Exception(f"Unable to delete resource at {path}: {resp.text}")
