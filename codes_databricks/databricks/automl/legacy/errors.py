class AutomlError(Exception):
    """
    Base exception class for AutoML errors.
    """

    def __init__(self, message="", *args):
        self.message = message
        super().__init__(message, *args)


class ExecutionTimeoutError(AutomlError):
    """
    User provided an insufficient timeout for notebook/trial execution
    """
    pass


class ExecutionResultError(AutomlError):
    """
    No results returned by hyperopt.
    """
    pass


class InvalidArgumentError(AutomlError):
    """
    Invalid argument set by user.
    """
    pass


class UnsupportedParameterError(AutomlError):
    """
    This parameter is not supported
    """
    pass


class UnsupportedDataError(AutomlError):
    """
    Input DataFrame has attributes that are unsupported.
    """
    pass


class UnsupportedColumnError(AutomlError):
    """
    Column has attributes that are unsupported.
    """
    pass


class UnsupportedRuntimeError(AutomlError):
    """
    User attempted to run AutoML on a runtime version that is not supported.
    """
    pass


class UnsupportedClusterError(AutomlError):
    """
    User attempted to run AutoML on a cluster that is not supported
    """
    pass


class ExperimentInitializationError(AutomlError):
    """
    Experiment was not initialized before use.
    """
    pass


class TrialFailedError(AutomlError):
    """
    Trail execution failed
    """
    pass


class NotebookImportSizeExceededError(AutomlError):
    """
    The notebook we are trying to import exceeded in size
    """
    pass


class InvalidSectionInputError(AutomlError):
    """
    Input to a section is invalid, such as empty input list for boolean preprocessor
    """
    pass


class ExperimentDirectoryDoesNotExist(AutomlError):
    """
    Error to indicate that the default experiment directory picked by AutoML does not exist
    """
    pass
