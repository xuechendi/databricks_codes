from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject


class Notebook(_FeatureStoreObject):
    def __init__(
        self,
        notebook_id,
        revision_id,
        notebook_workspace_id,
        feature_table_workspace_id,
    ):
        self._notebook_id = notebook_id
        self._revision_id = revision_id
        self._notebook_workspace_id = notebook_workspace_id
        self._feature_table_workspace_id = feature_table_workspace_id

    @property
    def notebook_id(self):
        return self._notebook_id

    @property
    def revision_id(self):
        return self._revision_id

    @property
    def notebook_workspace_id(self):
        return self._notebook_workspace_id

    @property
    def feature_table_workspace_id(self):
        return self._feature_table_workspace_id

    @classmethod
    def from_proto(cls, notebook_proto):
        return cls(
            notebook_id=notebook_proto.notebook_id,
            revision_id=notebook_proto.revision_id,
            notebook_workspace_id=notebook_proto.notebook_workspace_id,
            feature_table_workspace_id=notebook_proto.feature_table_workspace_id,
        )
