from abc import ABC
from enum import Enum
from typing import Any, Dict, Optional, List

from databricks.automl.legacy.tags import Tag


class Severity(str, Enum):
    """
    Possible severities of alerts
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertDisplayGroup(Enum):
    SEMANTIC_TYPE = "semantic_type"
    OTHER = "other"


class AlertDisplay:
    def __init__(self, string: str, group: AlertDisplayGroup):
        """
        Wrapper object to store the string used to display the alert in the
        data exploration notebook

        :param string: String to display in the notebook
        :param group: Group under which the string is displayed
        """
        self.string = string
        self.group = group


class Alert(ABC):
    """
    Abstract base class for an AutoML alert. By default, all alerts have a name, severity and version. Some alerts may
    also override the "misc" property with any additional information that should be stored.
    """

    # constants for the key in the dictionary of alert properties
    NAME = "name"
    SEVERITY = "severity"
    VERSION = "version"
    MISC = "misc"

    COL_NAME = "id"
    COL_TYPE = "type"
    AFFECTED_IDS = "affected"
    VALUES = "values"
    OTHERS = "others"  # number of columns/identities/other things not stored since we store a max of MAX_VALUES_PER_KEY

    ADDITIONAL_INFO = "additional_info"  # additional information to display for alerts
    ADDITIONAL_INFO_KEY = "key"
    ADDITIONAL_INFO_VALUE = "value"

    MAX_VALUE_LENGTH = 14  # max half length of a value for mlflow logging
    MAX_VALUES_PER_KEY = 5

    def __init__(self,
                 name: Tag,
                 severity: Severity,
                 version: int,
                 additional_info: Dict[str, Any] = None):
        self.name = name
        self.severity = severity
        self.version = version
        self._additional_info = additional_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, Alert):
            return False
        for field in [Alert.SEVERITY, Alert.VERSION, Alert.MISC, Alert.NAME]:
            if getattr(self, field) != getattr(other, field):
                return False
        if self.get_additional_info() != other.get_additional_info():
            return False
        return True

    @property
    def misc(self) -> Dict[str, Any]:
        return {}

    def display(self) -> Optional[List[AlertDisplay]]:
        """
        Returns optional object used to display this alert on the data exploration notebook
        """
        return None

    def get_additional_info(self) -> Dict[str, List[Dict[str, str]]]:
        additional_info_dict = {}

        if self._additional_info:
            additional_info_dict = {
                Alert.ADDITIONAL_INFO: [{
                    Alert.ADDITIONAL_INFO_KEY: str(k),
                    Alert.ADDITIONAL_INFO_VALUE: str(v)
                } for k, v in self._additional_info.items()]
            }

        return additional_info_dict
