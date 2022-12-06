from databricks.automl import internal, legacy
from databricks.automl.shared import utils as shared_utils
"""
Currently, the both automl/service/automl/resources/notebooks/AutoMLDriverNotebook.py and
webapp/web/js/mlflow/autoML/AutoMLDriverNotebook.py call this file directly.

We'd like to be able to modify the fit(...) internal API more easily, so in ML-25324,
we will change automl/service/automl/resources/notebooks/AutoMLDriverNotebook.py to call
databricks.automl.internal.classify(...) instead.

After ML-25324, this file should ONLY be called by webapp/web/js/mlflow/autoML/AutoMLDriverNotebook.py,
so it only needs to call the legacy code.
"""


class Regressor:
    def __init__(self, context_type):
        self._context_type = context_type

    def fit(self, **kwargs):
        if shared_utils.is_automl_service_enabled():
            return internal.regressor.Regressor(context_type=self._context_type).fit(**kwargs)
        else:
            return legacy.regressor.Regressor(context_type=self._context_type).fit(**kwargs)
