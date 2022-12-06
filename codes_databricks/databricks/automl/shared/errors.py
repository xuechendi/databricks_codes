class AutomlError(Exception):
    """
    Base exception class for AutoML errors.
    """

    def __init__(self, message="", *args):
        self.message = message
        super().__init__(message, *args)


class InvalidArgumentError(AutomlError):
    """
    Invalid argument passed by the user
    """
    pass


class AutomlServiceError(AutomlError):
    """
    Error when calling the AutoML service
    """
    pass


class FeatureStoreError(AutomlError):
    """
    Error when calling Feature Store
    """
    pass


class UnsupportedDataError(AutomlError):
    """
    Input DataFrame has attributes that are unsupported.
    """
    pass
