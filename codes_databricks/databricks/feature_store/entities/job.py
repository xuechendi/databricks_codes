from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject


class Job(_FeatureStoreObject):
    def __init__(self, job_id, run_id, job_workspace_id, feature_table_workspace_id):
        self._job_id = job_id
        self._run_id = run_id
        self._job_workspace_id = job_workspace_id
        self._feature_table_workspace_id = feature_table_workspace_id

    @property
    def job_id(self):
        return self._job_id

    @property
    def run_id(self):
        return self._run_id

    @property
    def job_workspace_id(self):
        return self._job_workspace_id

    @property
    def feature_table_workspace_id(self):
        return self._feature_table_workspace_id

    @classmethod
    def from_proto(cls, job_proto):
        return cls(
            job_id=job_proto.job_id,
            run_id=job_proto.run_id,
            job_workspace_id=job_proto.job_workspace_id,
            feature_table_workspace_id=job_proto.feature_table_workspace_id,
        )
