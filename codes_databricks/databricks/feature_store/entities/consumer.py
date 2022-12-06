from databricks.feature_store.entities._feature_store_object import _FeatureStoreObject
from databricks.feature_store.entities.job import Job
from databricks.feature_store.entities.notebook import Notebook


class Consumer(_FeatureStoreObject):
    def __init__(self, features, notebook=None, job_run=None):
        self._features = features
        self._notebook = notebook
        self._job_run = job_run

    @property
    def features(self):
        return self._features

    @property
    def notebook(self):
        return self._notebook

    @property
    def job_run(self):
        return self._job_run

    @classmethod
    def from_proto(cls, consumer_proto):
        if consumer_proto.HasField("notebook"):
            return cls(
                features=list(consumer_proto.features),
                notebook=Notebook.from_proto(consumer_proto.notebook),
            )
        elif consumer_proto.HasField("job_run"):
            return cls(
                features=list(consumer_proto.features),
                job_run=Job.from_proto(consumer_proto.job_run),
            )
