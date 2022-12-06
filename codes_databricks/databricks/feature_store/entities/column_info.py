import abc


class ColumnInfo:
    def __eq__(cls, other):
        if not isinstance(other, cls.__class__):
            return False
        return cls.__dict__ == other.__dict__

    @property
    @abc.abstractmethod
    def output_name(cls):
        pass
