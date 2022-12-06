# Copyright 2018 Databricks, Inc.
#
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
# pylint: disable=logging-format-interpolation
# pylint: disable=invalid-name

import time
from tensorflow import keras

from sparkdl.horovod import log_to_driver

__all__ = ["LogCallback"]


def _stream_log(line):
    timestamp = time.time()
    log_line = f"{timestamp}: {line}"
    log_to_driver(log_line)


class LogCallback(keras.callbacks.Callback):
    """
    A simple HorovodRunner log callback that streams event logs to notebook cell output.
    """

    def __init__(self, per_batch_log=False):
        """
        :param per_batch_log: whether to output logs per batch, default: False.
        """
        super(LogCallback, self).__init__()
        self.current_epoch = "unknown"
        self.per_batch_log = per_batch_log

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        _stream_log(f"begin epoch: {epoch} begin, logs: {logs}")

    def on_batch_end(self, batch, logs=None):
        if self.per_batch_log:
            _stream_log(f"epoch: {self.current_epoch}, batch: {batch}, logs: {logs}")

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=no-self-use
        _stream_log(f"end epoch: {epoch}, logs: {logs}")
