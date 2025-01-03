import abc
import types
from typing import Any, Dict

import numpy as np
from matplotlib import pyplot as plt

from sympy import Symbol


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


def plot_pair(*, states, expected_states, arglist, x_name, y_name, file_id):
    x = Symbol(x_name)
    y = Symbol(y_name)

    plt.plot(
        states[:, arglist.index(x)],
        states[:, arglist.index(y)],
        label="model",
    )
    plt.plot(
        expected_states[:, arglist.index(x)],
        expected_states[:, arglist.index(y)],
        label="reference",
        alpha=0.5,
    )
    plt.scatter(
        states[:, arglist.index(x)],
        states[:, arglist.index(y)],
        label="model",
    )
    plt.scatter(
        expected_states[:, arglist.index(x)],
        expected_states[:, arglist.index(y)],
        label="reference",
        alpha=0.5,
    )
    plt.axis("equal")
    plt.legend()
    plt.savefig(f"{file_id}.png")
    plt.close()
    print(f"Write Pair {file_id}.png")


def plot_quaternion_timeseries(
    *, times, states, expected_states, arglist, x_name, file_id
):
    print(x_name + "w")
    w = Symbol(x_name + "w")
    x = Symbol(x_name + "x")
    y = Symbol(x_name + "y")
    z = Symbol(x_name + "z")

    for symbol in [w, x, y, z]:
        plt.plot(
            times,
            states[:, arglist.index(symbol)],
            label=f"model {symbol}",
        )
        plt.plot(
            times,
            expected_states[:, arglist.index(symbol)],
            label=f"reference {symbol}",
            alpha=0.5,
        )
    # plt.scatter(
    #     times,
    #     states[:, arglist.index(x)],
    #     label="model",
    # )
    # plt.scatter(
    #     times,
    #     expected_states[:, arglist.index(x)],
    #     label="reference",
    #     alpha=0.5,
    # )
    plt.legend()
    plt.savefig(f"{file_id}.png")
    print(f"Write Timeseries {file_id}.png")
