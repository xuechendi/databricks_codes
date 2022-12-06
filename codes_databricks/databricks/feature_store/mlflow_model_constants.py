# Shared mlflow model constants that are used in both the Feature Store client and lookup client

# Module name of the original mlflow_model
MLFLOW_MODEL_NAME = "databricks.feature_store.mlflow_model"

# FeatureStoreClient.log_model will log models containing a 'raw_model' folder within the data_path.
# This folder stores the MLmodel for the raw model, which is needed to run inference.
# FeatureStoreClient.log_model may be called by any version of the FeatureStoreClient >= 0.3.0,
# however this module may be from a more recent version. As such:
# *** The value of this constant SHOULD NEVER CHANGE ***
RAW_MODEL_FOLDER = "raw_model"

# Constant for the ML model filename
ML_MODEL = "MLmodel"

# The package name of the feature lookup client as published on PyPI
FEATURE_LOOKUP_CLIENT_PIP_PACKAGE = "databricks-feature-lookup"

# The major version of the feature lookup client.  This is a shared constant in order to tightly couple:
#
# - The Feature lookup client major version (eg `{FEATURE_LOOKUP_CLIENT_PIP_PACKAGE}.0.1`)
# - The pinned major version of the databricks-feature-lookup requirement in logged model's conda.yaml
#
# If needed, these can be decoupled into separate constants, but that decision should be carefully considered.
FEATURE_LOOKUP_CLIENT_MAJOR_VERSION = 0

DATABRICKS = "Databricks"
SAGEMAKER = "SageMaker"
