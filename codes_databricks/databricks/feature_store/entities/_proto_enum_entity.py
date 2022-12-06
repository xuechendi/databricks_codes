import abc


class _ProtoEnumEntity(object):
    """
    Generic entity to map to proto enum messages.
    When inheriting from this class override `_enum_type` method to return underlying enum proto.
    """

    _STRING_TO_ENUM, _ENUM_TO_STRING = None, None

    @classmethod
    def init(cls):
        if not cls._STRING_TO_ENUM:
            enum_proto = cls._enum_type()
            cls._STRING_TO_ENUM = {k: enum_proto.Value(k) for k in enum_proto.keys()}
            cls._ENUM_TO_STRING = {
                value: key for key, value in cls._STRING_TO_ENUM.items()
            }

    @classmethod
    @abc.abstractmethod
    def _enum_type(cls) -> str:
        pass

    @classmethod
    def from_string(cls, str_value):
        cls.init()
        if str_value.upper() not in cls._STRING_TO_ENUM:
            raise Exception(
                f"Could not find a {cls._enum_type().DESCRIPTOR.name} value corresponding to the "
                f" input string '{str_value}'. Valid values: '{list(cls._STRING_TO_ENUM.keys())}'"
            )
        return cls._STRING_TO_ENUM[str_value.upper()]

    @classmethod
    def to_string(cls, value):
        cls.init()
        if value not in cls._ENUM_TO_STRING:
            raise Exception(
                f"Input {cls._enum_type().DESCRIPTOR.name} type '{value}' cannot be converted to"
                f" a string representation. Valid inputs: '{list(cls._ENUM_TO_STRING.keys())}'"
            )
        return cls._ENUM_TO_STRING[value]
