import abc
import types
from itertools import product
from typing import Dict, Tuple, Union

import numpy as np
from formak.exceptions import ModelConstructionError
from sympy import Symbol, diff
from sympy.solvers.solveset import nonlinsolve


class UiModelBase:
    """
    Use as a base class for ui.Model, but in a separate file from ui.Model so that formak.python doesn't directly depend on formak.ui just for typing
    """

    pass


def model_validation(
    state_model,
    process_noise: Dict[Union[Symbol, Tuple[Symbol, Symbol]], float],
    sensor_models,
    *,
    verbose=True,
    extra_validation=False,
    calibration_map: Dict[Symbol, float],
):
    assert isinstance(process_noise, dict)
    allowed_keys = set(
        list(state_model.control)
        + [
            (x, y)
            for x, y in product(state_model.control, state_model.control)
            if x != y
        ]
    )
    for key in process_noise:
        if not isinstance(key, Symbol):
            raise ModelConstructionError(
                f"Key {key} needs to be type Symbol, found type {type(key)}"
            )
        if key not in allowed_keys:
            render = ", ".join([f"{k} {type(k)}" for k in allowed_keys])
            raise ModelConstructionError(
                f'Key {key} {type(key)} not in allow"list" of keys and combinations for process noise based on state_model.control [{render}]'
            )
    if set(calibration_map.keys()) != state_model.calibration:
        map_version = set(calibration_map.keys())
        missing_calibrations = state_model.calibration - map_version
        missing_calibrations = ", ".join(
            sorted([symbol.name for symbol in missing_calibrations])[:3]
        )
        extra_mappings = map_version - state_model.calibration
        extra_mappings = ", ".join(
            sorted([symbol.name for symbol in extra_mappings])[:3]
        )
        raise ModelConstructionError(
            f"Mismatch in Model calibration: Missing from map? {missing_calibrations} | Missing from setup? {extra_mappings}"
        )

    # Check if sensor models depend on values outside the state, calibration [and map]
    allowed_symbols = set(state_model.state) | set(state_model.calibration)
    for k, model_set in sensor_models.items():
        for k2, model in model_set.items():
            if not set(model.free_symbols).issubset(allowed_symbols):
                extra_symbols = sorted(list(set(model.free_symbols) - allowed_symbols))
                raise ModelConstructionError(
                    f"Sensor Model[{k}][{k2}] has symbols not in state, calibration: {extra_symbols}"
                )

    # Note: flagging this off for now because it was running into performance issues
    if extra_validation:
        symbols_to_solve_for = list(state_model.state)
        equations_to_solve = [
            diff(model, symbol)
            for model, symbol in product(
                state_model.state_model.values(), state_model.state
            )
        ]
        results_set = nonlinsolve(equations_to_solve, symbols_to_solve_for)
        if len(results_set) > 0:
            solutions = [
                dict(zip(symbols_to_solve_for, solution))
                for solution in sorted(list(results_set))
            ]
            solution_repr = "\n - ".join(map(lambda x: str(x), solutions[:3]))
            if verbose:
                print("symbols_to_solve_for")
                print(symbols_to_solve_for)
                print("equations_to_solve")
                for e in equations_to_solve:
                    print(f" - {e}")
            raise ModelConstructionError(
                f"Model has solutions in state space where covariance will collapse to zero. Example Solutions:\n -{solution_repr}"
            )


class _NamedArrayBase(abc.ABC):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        kwargs = ", ".join(f"{k}={v}" for k, v in self._kwargs.items())
        return f"{self.name}({kwargs})"

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def shape(cls):
        raise NotImplementedError()

    @classmethod
    def from_data(cls, data):
        assert data.shape == cls.shape()
        return cls(_data=data)


def named_vector(name, arglist):
    class _NamedVector(_NamedArrayBase):
        def __init__(self, *, _data=None, **kwargs):
            super().__init__(name)
            self._kwargs = kwargs

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
                    self.data[idx, 0] = kwargs[key]

        @classmethod
        def __subclasshook__(cls, Other):
            print(Other, Other.__name__)
            print(Other.arglist)
            1 / 0
            return (
                Other.__name__ == name
                and arglist == Other.arglist
                and cls.shape() == Other.shape()
            )

        @classmethod
        def shape(cls):
            return (len(arglist), 1)

    return types.new_class(name, bases=(_NamedVector,))


def named_covariance(name, arglist):
    class _NamedCovariance(_NamedArrayBase):
        def __init__(self, *, _data=None, **kwargs):
            super().__init__(name)
            self._kwargs = kwargs

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
            print(Other, Other.__name__)
            print(Other.arglist)
            1 / 0
            return (
                Other.__name__ == name
                and arglist == Other.arglist
                and cls.shape() == Other.shape()
            )

        @classmethod
        def shape(cls):
            return (len(arglist), len(arglist))

    return types.new_class(name, bases=(_NamedCovariance,))
