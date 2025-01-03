import types

import numpy as np
from backend_py.named_array_base import _NamedArrayBase


def named_covariance(name, arglist):
    class _NamedCovariance(_NamedArrayBase):
        _name = name
        _arglist = arglist
        shape = (len(arglist), len(arglist))

        def __init__(self, *, _data=None, **kwargs):
            super().__init__(name, kwargs)

            allowed_keys = [str(arg) for arg in arglist]
            for key in kwargs:
                if key not in allowed_keys:
                    raise TypeError(
                        f"{name}() got an unexpected keyword argument {key}"
                    )

            if _data is not None:
                assert len(kwargs) == 0
                self.data = _data
            else:
                self.data = np.eye(len(arglist))

            for idx, key in enumerate(allowed_keys):
                if key in kwargs:
                    self.data[idx, idx] = kwargs[key]

        @classmethod
        def __subclasshook__(cls, Other):
            return (
                Other.__name__ == name
                and cls._arglist == Other._arglist
                and cls.shape == Other.shape
            )

    return types.new_class(name, bases=(_NamedCovariance,))
