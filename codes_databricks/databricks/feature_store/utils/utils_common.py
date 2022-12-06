from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository


def is_artifact_uri(uri):
    """
    Checks the artifact URI is associated with a MLflow model or run.
    The actual URI can be a model URI, model URI + subdirectory, or model URI + path to artifact file.
    """
    return ModelsArtifactRepository.is_models_uri(
        uri
    ) or RunsArtifactRepository.is_runs_uri(uri)
