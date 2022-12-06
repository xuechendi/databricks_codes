import pprint


class _FeatureStoreObject(object):
    def __iter__(self):
        # Iterate through list of properties and yield as key -> value
        for prop in self._properties():
            yield prop, self.__getattribute__(prop)

    @classmethod
    def _properties(cls):
        return sorted(
            [p for p in cls.__dict__ if isinstance(getattr(cls, p), property)]
        )

    @classmethod
    def from_dictionary(cls, the_dict):
        filtered_dict = {
            key: value for key, value in the_dict.items() if key in cls._properties()
        }
        return cls(**filtered_dict)

    def __repr__(self):
        return _EntityPrinter().to_string(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__


class _EntityPrinter(object):
    def __init__(self):
        super().__init__()
        self.printer = pprint.PrettyPrinter(indent=1)

    def to_string(self, obj):
        if isinstance(obj, _FeatureStoreObject):
            return f"<{type(obj).__name__}: {self._entity_to_string(obj)}>"
        return self.printer.pformat(obj)

    def _entity_to_string(self, entity):
        return ", ".join([f"{key}={self.to_string(value)}" for key, value in entity])
