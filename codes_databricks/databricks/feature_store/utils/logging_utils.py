import sys
import logging
import logging.config


# Logging format example:
# 2020/10/02 18:33:32 INFO databricks.feature_store.client: The feature table ...
LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


def _configure_feature_store_loggers(root_module_name):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "feature_store_formatter": {
                    "format": LOGGING_LINE_FORMAT,
                    "datefmt": LOGGING_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "feature_store_handler": {
                    "level": "INFO",
                    "formatter": "feature_store_formatter",
                    "class": "logging.StreamHandler",
                    "stream": sys.stderr,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["feature_store_handler"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )
