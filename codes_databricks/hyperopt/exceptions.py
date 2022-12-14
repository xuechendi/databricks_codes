"""
"""


class BadSearchSpace(Exception):
    """Something is wrong in the description of the search space"""


class BadObjectiveFunction(Exception):
    """Something is wrong with the objective function Hyperopt is optimizing"""


class DuplicateLabel(BadSearchSpace):
    """A search space included a duplicate label"""


class InvalidTrial(ValueError):
    """Non trial-like object used as Trial"""

    def __init__(self, msg, obj):
        ValueError.__init__(self, msg + " " + str(obj))
        self.obj = obj


class InvalidResultStatus(ValueError):
    """Status of fmin evaluation was not in base.STATUS_STRINGS"""

    def __init__(self, result):
        ValueError.__init__(self)
        self.result = result


class InvalidLoss(ValueError):
    """fmin evaluation returned invalid loss value"""

    def __init__(self, result):
        ValueError.__init__(self)
        self.result = result


class AllTrialsFailed(Exception):
    """All optimization steps have finished with status base.STATUS_FAIL"""


class InvalidAnnotatedParameter(ValueError):
    """fn has a type hint that is not from hp."""

    def __init__(self, result):
        ValueError.__init__(self)
        self.result = result


# -- flake8 doesn't like blank last line
