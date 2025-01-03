import types

import numpy as np
from backend_py.named_array_base import _NamedArrayBase


def named_vector(name, arglist):
    class _NamedVector(_NamedArrayBase):
        _name = name
        _arglist = arglist
        shape = (len(arglist), 1)

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
                self.data = np.zeros((len(arglist), 1))

            for idx, key in enumerate(allowed_keys):
                if key in kwargs:
                    val = kwargs[key]
                    self.data[idx, 0] = val

        @classmethod
        def __subclasshook__(cls, Other):
            return (
                Other.__name__ == name
                and cls._arglist == Other._arglist
                and cls.shape == Other.shape
            )

        def render_diff(self, expected_state):
            def diff_source():
                for key, model, expected in zip(
                    self._arglist, self.data, expected_state.data
                ):
                    float(model)
                    float(expected)
                    if np.allclose([model], [expected]):
                        continue
                    yield key, model, expected, model - expected

            print(
                "Key".ljust(30)
                + "|"
                + "Model".ljust(15)
                + "|"
                + "Expected".ljust(15)
                + "|"
                + "Diff".ljust(15)
            )
            for key, model, expected, delta in sorted(
                list(diff_source()), key=lambda row: abs(row[-1]), reverse=True
            ):
                key = str(key)[:30].ljust(30)
                model = f"{model[0]: >15.9g}".rjust(15)
                print(f"{key}|{model}|{expected[0]: >15.9g}|{delta[0]: >15.9g}")

    return types.new_class(name, bases=(_NamedVector,))
