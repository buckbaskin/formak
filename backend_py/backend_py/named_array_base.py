import abc
from typing import Any, Dict


class _NamedArrayBase(abc.ABC):
    def __init__(self, name: str, kwargs: Dict[Any, Any]):
        self.name = name
        self._kwargs = kwargs
        self.data = None  # type: Optional[NDArray]

    def __repr__(self):
        kwargs = ", ".join(
            f"{k}={float(v)}" for k, v in sorted(list(self._kwargs.items()))
        )
        return f"{self.name}({kwargs})"

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def from_data(cls, data):
        if data.shape != cls.shape:
            raise ValueError(f"Expected shape {cls.shape}, got shape {data.shape}")
        return cls(_data=data)

    @classmethod
    def from_dict(cls, mapping):
        return cls(**{str(k): v for k, v in mapping.items()})

    @classmethod
    def __subclasshook__(cls, Other):
        raise NotImplementedError()
