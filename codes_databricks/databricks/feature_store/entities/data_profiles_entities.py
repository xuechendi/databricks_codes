"""Entities definition for data profiles module."""

from enum import Enum


class ComputationMode(str, Enum):
    MANUAL = "MANUAL"
    AUTO = "AUTO"


class Granularities(str, Enum):
    GLOBAL = "GLOBAL"
    ONE_HOUR = "1 HOUR"
    ONE_DAY = "1 DAY"
    ONE_WEEK = "1 WEEK"
    ONE_MONTH = "1 MONTH"
