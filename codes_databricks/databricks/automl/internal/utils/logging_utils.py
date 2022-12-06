import logging
import logging.config
import sys

# Logging format example:
# 2022/02/22 12:36:37 INFO databricks.automl.supervised_learner: Started new AutoML run

LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


def _configure_automl_loggers(root_module_name):
    """
    This function must be called exactly once from, from automl/python/databricks/automl/__init__.py
    If you call it more than once, or call it in other files, automl logging may break and unit tests may fail
    """
    logging.config.dictConfig({
        "version": 1,
        "formatters": {
            "automl_formatter": {
                "format": LOGGING_LINE_FORMAT,
                "datefmt": LOGGING_DATETIME_FORMAT,
            },
        },
        "handlers": {
            "automl_handler": {
                "class": "logging.StreamHandler",
                "formatter": "automl_formatter",
            },
        },
        "loggers": {
            root_module_name: {
                "handlers": ["automl_handler"],
                "level": "INFO",
                "propagate": False,
            }
        }
    })
