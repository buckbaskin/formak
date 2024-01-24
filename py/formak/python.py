from __future__ import annotations

import dataclasses
from collections import namedtuple
from dataclasses import dataclass
from itertools import count
from math import sqrt
from typing import Any, Dict

import numpy as np
from formak.exceptions import MinimizationFailure, ModelConstructionError
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sympy import Matrix, Symbol, cse, simplify
from sympy.utilities.lambdify import lambdify

from formak import common

DEFAULT_MODULES = ("scipy", "numpy", "math", {"sec": lambda v: 1.0 / np.cos(v)})


@dataclass(frozen=True)
class Config:
    """
    Options for generating C++.

    common_subexpression_elimination:
        Remove common shared computation
    python_modules:
        Allow dependencies. Math is the Python standard library
    extra_validation:
        Catch errors earlier in exchange for increased compute time
    """

    common_subexpression_elimination: bool = True
    python_modules: tuple[Any, Any, Any, Any] = DEFAULT_MODULES
    extra_validation: bool = False
    max_dt_sec: float = 0.1
    innovation_filtering: float | None = 5.0


class BasicBlock:
    """
    A run of statements without control flow.

    All statements can be reordered or changed to improve performance.
    """

    def __init__(self, *, arglist: list[str], statements: list[Any], config: Config):
        self._arglist = arglist
        self._exprs = statements
        self._config = config

        self._compile()

    def __len__(self):
        return len(self._exprs)

    def _compile(self):
        prefix = []
        body = self._exprs

        if self._config.common_subexpression_elimination:
            prefix, body = cse(body, symbols=(Symbol(f"_t{i}") for i in count()))

        temporaries = [r[0] for r in prefix]
        self._prefix = []
        for i in range(len(prefix)):
            expr = prefix[i][1]
            if self._config.common_subexpression_elimination:
                expr = simplify(expr)

            self._prefix.append(
                (
                    temporaries[i],
                    lambdify(
                        self._arglist + temporaries[:i],
                        expr,
                        modules=self._config.python_modules,
                        cse=False,
                    ),
                )
            )

        self._body = [
            lambdify(
                self._arglist + temporaries,
                simplify(expr)
                if self._config.common_subexpression_elimination
                else expr,
                modules=self._config.python_modules,
                cse=False,
            )
            for expr in body
        ]

    def execute(self, *args, **kwargs):

        # Note: The list of statements is ordered and can get CSE or reordered within the block because we know it is straight calculation without control flow (a basic block)
        temporary_values = {}
        for name, expr in self._prefix:
            temporary_values[str(name)] = expr(*args, **kwargs, **temporary_values)

        for impl in self._body:
            yield impl(*args, **kwargs, **temporary_values)


class Model:
    """Python implementation of the model."""

    def __init__(self, symbolic_model, config, calibration_map=None):
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)

        if calibration_map is None:
            calibration_map = {}

        self.state_size = len(symbolic_model.state)
        self.calibration_size = len(symbolic_model.calibration)
        self.control_size = len(symbolic_model.control)

        self.arglist_state = sorted(list(symbolic_model.state), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(symbolic_model.calibration), key=lambda x: x.name
        )
        self.arglist_control = sorted(
            list(symbolic_model.control), key=lambda x: x.name
        )
        self.arglist = (
            [symbolic_model.dt]
            + self.arglist_state
            + self.arglist_calibration
            + self.arglist_control
        )

        self.State = common.named_vector("State", self.arglist_state)
        self.Control = common.named_vector("Control", self.arglist_control)
        self.Calibration = common.named_vector("Calibration", self.arglist_calibration)

        self.calibration_vector = np.zeros((0, 0))
        if self.calibration_size > 0:
            if len(calibration_map) == 0:
                map_lite = ", ".join(
                    [f"{k}: ..." for k in self.arglist_calibration[:3]]
                )
                if len(self.arglist_calibration) > 3:
                    map_lite += ", ..."
                raise ModelConstructionError(
                    f"Model Missing specification of calibration_map: {{{map_lite}}}"
                )
            if len(calibration_map) != self.calibration_size:
                missing_from_map = set(symbolic_model.calibration) - set(
                    calibration_map.keys()
                )
                extra_from_map = set(calibration_map.keys()) - set(
                    symbolic_model.calibration
                )
                missing = ""
                if len(missing_from_map) > 0:
                    missing = f"\nMissing: {missing_from_map}"
                extra = ""
                if len(extra_from_map) > 0:
                    extra = f"\nExtra: {extra_from_map}"
                raise ModelConstructionError(f"Mismatched Calibration:{missing}{extra}")
        self.calibration_vector = np.array(
            [[calibration_map[k] for k in self.arglist_calibration]]
        ).transpose()
        if self.calibration_vector.shape != (self.calibration_size, 1):
            raise ModelConstructionError(
                f"calibration_vector shape {self.calibration_vector.shape} doesn't match {(self.calibration_size, 1)}"
            )

        self._impl = BasicBlock(
            arglist=self.arglist,
            statements=[symbolic_model.state_model[a] for a in self.arglist_state],
            config=config,
        )

    def model(self, dt, state, control=None):
        if control is None:
            if self.control_size > 0:
                raise TypeError(
                    "model() missing 1 required positional argument: 'control'"
                )
            control = self.Control()

        try:
            assert isinstance(dt, float)
            assert isinstance(state, self.State)
            assert isinstance(control, self.Control)
        except AssertionError:
            print(
                f"model({type(dt)} {dt}, {type(state)} {state}, {type(control)} {control}"
            )
            raise

        next_state = self.State(
            **{
                str(state_id): result
                for state_id, result in zip(
                    self.arglist_state,
                    self._impl.execute(dt, *state, *self.calibration_vector, *control),
                )
            }
        )

        return next_state


class SensorModel:
    def __init__(self, state_model, sensor_model, calibration_map, config):
        self.readings = sorted(list(sensor_model.keys()))
        self.sensor_models = sensor_model

        self.sensor_size = len(self.readings)
        self.state_size = len(state_model.state)
        self.calibration_size = len(state_model.calibration)

        self.arglist_state = sorted(list(state_model.state), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(state_model.calibration), key=lambda x: x.name
        )
        self.arglist = self.arglist_state + self.arglist_calibration

        self.State = common.named_vector("State", self.arglist_state)
        self.Covariance = common.named_covariance("Covariance", self.arglist_state)
        self.Calibration = common.named_vector("Calibration", self.arglist_calibration)
        self.Reading = common.named_vector("Reading", self.readings)
        self.ReadingCovariance = common.named_vector("ReadingCovariance", self.readings)

        self.calibration_vector = np.array(
            [[calibration_map[k] for k in self.arglist_calibration]]
        ).transpose()
        if self.calibration_vector.shape != (self.calibration_size, 1):
            raise ModelConstructionError(
                f"calibration vector shape {self.calibration_vector.shape} doesn't match expected shape {(self.calibration_size, 1)}"
            )

        self._impl = BasicBlock(
            arglist=self.arglist,
            statements=[sensor_model[k] for k in self.readings],
            config=config,
        )

        ## "Pre-flight" Checks

        # Pre-check model for type errors
        self.model(self.State())

    def __len__(self):
        return len(self.sensor_models)

    def model(self, state_vector):
        assert isinstance(state_vector, self.State)

        reading = np.zeros((self.sensor_size, 1))
        for i, (reading_id, result) in enumerate(
            zip(
                self.readings,
                self._impl.execute(*state_vector, *self.calibration_vector),
            )
        ):
            try:
                reading[i, 0] = result
            except (TypeError, ValueError):
                print(
                    "Error when trying to process sensor model for reading %s"
                    % (reading_id,)
                )
                print("expected: float")
                print("given: {}, {}".format(state_vector, self.calibration_vector))
                if "result" in locals():
                    print("found: {}, {}".format(type(result), result))
                raise
        return self.Reading.from_data(reading)


StateAndCovariance = namedtuple("StateAndCovariance", ["state", "covariance"])


class ExtendedKalmanFilter:
    def __init__(
        self,
        state_model,
        process_noise: dict[Symbol | tuple[Symbol, Symbol], float],
        sensor_models,
        sensor_noises: dict[Symbol | tuple[Symbol, Symbol], float],
        config,
        calibration_map=None,
    ):
        if calibration_map is None:
            calibration_map = {}

        assert isinstance(config, Config)
        assert isinstance(process_noise, dict)
        assert isinstance(calibration_map, dict)

        self.config = config

        self.state_size = len(state_model.state)
        self.control_size = len(state_model.control)
        self.calibration_size = len(state_model.calibration)
        self.arglist_state = sorted(list(state_model.state), key=lambda x: x.name)
        self.arglist_control = sorted(list(state_model.control), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(state_model.calibration), key=lambda x: x.name
        )

        self.State = common.named_vector("State", self.arglist_state)
        self.Covariance = common.named_covariance("Covariance", self.arglist_state)
        self.Control = common.named_vector("Control", self.arglist_control)
        self.Calibration = common.named_vector("Calibration", self.arglist_calibration)

        self.process_noise = None
        self.sensor_models = sensor_models
        self.sensor_noises = sensor_noises
        self.calibration_map = calibration_map

        self._construct_process(
            state_model=state_model,
            process_noise=process_noise,
            calibration_map=calibration_map,
            config=config,
        )
        self._construct_sensors(
            state_model=state_model,
            sensor_models=sensor_models,
            sensor_noises=sensor_noises,
            calibration_map=calibration_map,
            config=config,
        )

    def _construct_process(self, state_model, process_noise, calibration_map, config):
        self._state_model = Model(
            symbolic_model=state_model, calibration_map=calibration_map, config=config
        )
        assert len(process_noise) == self.control_size

        self.calibration_vector = self._state_model.calibration_vector

        process_noise_matrix = np.eye(self._state_model.control_size)

        for iIdx, iSymbol in enumerate(self.arglist_control):
            for jIdx, jSymbol in enumerate(self.arglist_control):
                if (iSymbol, jSymbol) in process_noise:
                    value = process_noise[(iSymbol, jSymbol)]
                elif (jSymbol, iSymbol) in process_noise:
                    value = process_noise[(jSymbol, iSymbol)]
                elif iSymbol == jSymbol and iSymbol in process_noise:
                    value = process_noise[iSymbol]
                else:
                    value = 0.0
                process_noise_matrix[iIdx, jIdx] = value
                process_noise_matrix[jIdx, iIdx] = value

        self.process_noise = process_noise_matrix

        # TODO(buck): Reorder state vector (arglist*) to take advantage of sparse blocks (e.g. assign in a block, skip a block, etc)

        process_matrix = Matrix(
            [state_model.state_model[a] for a in self._state_model.arglist_state]
        )
        symbolic_process_jacobian = process_matrix.jacobian(
            self._state_model.arglist_state
        )
        # TODO(buck): This assertion won't necessarily hold if CSE is on across states
        assert symbolic_process_jacobian.shape == (
            self.state_size,
            self.state_size,
        )

        symbolic_control_jacobian = []
        if self.control_size > 0:
            symbolic_control_jacobian = process_matrix.jacobian(self.arglist_control)

        self._impl_process_jacobian = BasicBlock(
            arglist=self._state_model.arglist,
            # Flatten the Matrix symbolic_process_jacobian by iterating over all elements
            statements=[expr for expr in symbolic_process_jacobian],
            config=config,
        )
        assert len(self._impl_process_jacobian) == self.state_size**2

        self._impl_control_jacobian = BasicBlock(
            arglist=self._state_model.arglist,
            statements=[expr for expr in symbolic_control_jacobian],
            config=config,
        )
        assert len(self._impl_control_jacobian) == self.control_size * self.state_size

    def _construct_sensors(
        self, state_model, sensor_models, sensor_noises, calibration_map, config: Config
    ):
        assert set(sensor_models.keys()) == set(sensor_noises.keys())
        assert isinstance(sensor_noises, dict)
        assert len(sensor_noises) == len(sensor_models)

        self.sensor_models = {
            k: SensorModel(
                state_model=state_model,
                sensor_model=model,
                calibration_map=calibration_map,
                config=config,
            )
            for k, model in sensor_models.items()
        }

        matrix_sensor_noises = {}
        for key, model in self.sensor_models.items():
            assert isinstance(sensor_noises[key], dict)
            assert len(sensor_noises[key]) == self.sensor_models[key].sensor_size

            matrix_sensor_noises[key] = model.ReadingCovariance.from_dict(
                sensor_noises[key]
            )

        self.sensor_noises = matrix_sensor_noises  # type: Dict[str, NDArray]

        self.arglist_sensor = self.arglist_state + self.arglist_calibration

        self._impl_sensor_jacobians = {}

        for k, sensor_model in self.sensor_models.items():
            sensor_size = len(sensor_model.readings)

            sensor_matrix = Matrix(
                [sensor_model.sensor_models[r] for r in sensor_model.readings]
            )
            symbolic_sensor_jacobian = sensor_matrix.jacobian(self.arglist_sensor)
            # TODO(buck): This assertion won't necessarily hold if CSE is on across states
            assert symbolic_sensor_jacobian.shape == (
                sensor_size,
                self.state_size + self.calibration_size,
            )

            impl_sensor_jacobian = BasicBlock(
                arglist=self.arglist_sensor,
                statements=[expr for expr in symbolic_sensor_jacobian],
                config=config,
            )
            assert len(impl_sensor_jacobian) == sensor_size * (
                self.state_size + self.calibration_size
            )

            # TODO(buck): allow for compiling only process, sensors or list of specific sensors
            self._impl_sensor_jacobians[k] = impl_sensor_jacobian

        self.innovations = {}  # type: Dict[str, NDArray]
        self.sensor_prediction_uncertainty = {}  # type: Dict[str, NDArray]

    def make_reading(self, key, *, data=None, **kwargs):
        if len(kwargs) == 0 and data is not None:
            return self.sensor_models[key].Reading.from_data(data)

        return self.sensor_models[key].Reading(**kwargs)

    def process_jacobian(self, dt, state, control):
        computed_jacobian = list(
            self._impl_process_jacobian.execute(
                dt, *state, *self.calibration_vector, *control
            )
        )

        jacobian = np.zeros((self.state_size, self.state_size))
        for row in range(self.state_size):
            for col in range(self.state_size):
                result = computed_jacobian[row * self.state_size + col]
                jacobian[row, col] = result
        return jacobian

    def control_jacobian(self, dt, state, control):
        computed_jacobian = list(
            self._impl_control_jacobian.execute(
                dt, *state, *self.calibration_vector, *control
            )
        )
        result = np.zeros((self.state_size, self.control_size))
        for row in range(self.state_size):
            for col in range(self.control_size):
                result[row, col] = computed_jacobian[row * self.control_size + col]
        return result

    def sensor_jacobian(self, sensor_key, state):
        sensor_size = self.sensor_models[sensor_key].sensor_size

        impl_sensor_jacobian = self._impl_sensor_jacobians[sensor_key]

        computed_jacobian = list(
            impl_sensor_jacobian.execute(*state, *self.calibration_vector)
        )
        result = np.zeros((sensor_size, self.state_size))
        for row in range(sensor_size):
            for col in range(self.state_size):
                result[row, col] = computed_jacobian[row * sensor_size + col]
        return result

    def process_model(self, dt, state, covariance, control=None):
        if control is None:
            control = self.Control()

        try:
            assert isinstance(state, self.State)
            assert isinstance(covariance, self.Covariance)
            assert isinstance(control, self.Control)
        except AssertionError:
            print(
                "process_model(dt: %s, state: %s, covariance: %s, control: %s)"
                % (type(dt), type(state), type(covariance), type(control))
            )
            raise

        # TODO(buck): CSE across the whole process computation (model, jacobians)
        G_t = self.process_jacobian(dt, state, control)
        V_t = self.control_jacobian(dt, state, control)

        next_state_covariance = np.matmul(
            G_t, np.matmul(covariance.data, G_t.transpose())
        )
        assert next_state_covariance.shape == covariance.shape

        next_control_covariance = np.matmul(
            V_t, np.matmul(self.process_noise, V_t.transpose())
        )
        assert next_control_covariance.shape == covariance.shape

        next_covariance = next_state_covariance + next_control_covariance
        assert next_covariance.shape == covariance.shape

        next_state = self._state_model.model(dt, state, control)
        assert isinstance(next_state, self.State)

        return StateAndCovariance(
            next_state, self.Covariance.from_data(next_covariance)
        )

    def remove_innovation(self, innovation: NDArray, S_inv: NDArray) -> bool:
        if self.config.innovation_filtering is None:
            return False

        editing_threshold = self.config.innovation_filtering  # type: float
        normalized_innovation = innovation.transpose() * S_inv * innovation
        (sensor_size, _) = innovation.shape
        expected_innovation = editing_threshold * sqrt(2 * sensor_size) + sensor_size
        return normalized_innovation > expected_innovation

    def sensor_model(self, state, covariance, *, sensor_key, sensor_reading):
        model_impl = self.sensor_models[sensor_key]
        sensor_size = len(model_impl.readings)
        Q_t = _model_noise = self.sensor_noises[sensor_key]

        try:
            assert isinstance(state, self.State)
            assert isinstance(covariance, self.Covariance)
            assert isinstance(sensor_reading, model_impl.Reading)
        except AssertionError:
            print(
                "sensor_model(state: %s, covariance: %s, sensor_key: %s, sensor_reading: %s)"
                % (
                    type(state),
                    type(covariance),
                    type(sensor_key),
                    type(sensor_reading),
                )
            )
            raise

        expected_reading = model_impl.model(state)
        assert isinstance(expected_reading, model_impl.Reading)

        H_t = self.sensor_jacobian(sensor_key, state)
        assert H_t.shape == (sensor_size, self.state_size)

        self.sensor_prediction_uncertainty[sensor_key] = S_t = (
            np.matmul(H_t, np.matmul(covariance.data, H_t.transpose())) + Q_t.data
        )
        S_inv = np.linalg.inv(S_t)

        self.innovations[sensor_key] = innovation = (
            sensor_reading.data - expected_reading.data
        )

        if self.remove_innovation(innovation, S_inv):
            return StateAndCovariance(state, covariance)

        K_t = _kalman_gain = np.matmul(
            covariance.data, np.matmul(H_t.transpose(), S_inv)
        )

        next_covariance = covariance.data - np.matmul(
            K_t, np.matmul(H_t, covariance.data)
        )

        next_state = state.data + np.matmul(K_t, innovation)

        return StateAndCovariance(
            self.State.from_data(next_state), self.Covariance.from_data(next_covariance)
        )


def compile(symbolic_model, calibration_map=None, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    common.model_validation(
        symbolic_model,
        {},
        {},
        calibration_map=calibration_map,
        extra_validation=config.extra_validation,
    )

    return Model(
        symbolic_model=symbolic_model, calibration_map=calibration_map, config=config
    )


def compile_ekf(
    symbolic_model: common.UiModelBase,
    process_noise: dict[Symbol | tuple[Symbol, Symbol], float],
    sensor_models,
    sensor_noises,
    calibration_map=None,
    *,
    config=None,
) -> ExtendedKalmanFilter:
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    common.model_validation(
        symbolic_model,
        process_noise,
        sensor_models,
        calibration_map=calibration_map,
        extra_validation=config.extra_validation,
    )

    return ExtendedKalmanFilter(
        state_model=symbolic_model,
        process_noise=process_noise,
        sensor_models=sensor_models,
        sensor_noises=sensor_noises,
        calibration_map=calibration_map,
        config=config,
    )


def force_to_ndarray(mat: Any) -> NDArray | None:
    if mat is None:
        return mat

    if isinstance(mat, list):
        return np.array(mat)
    if not isinstance(mat, np.ndarray):
        mat = mat.__array__()

    assert isinstance(mat, np.ndarray)

    return mat


class SklearnEKFAdapter(BaseEstimator):
    allowed_keys = [
        "symbolic_model",
        "process_noise",
        "sensor_models",
        "sensor_noises",
        "calibration_map",
        "config",
    ]

    @classmethod
    def Create(
        cls,
        symbolic_model,
        process_noise: dict[Symbol | tuple[Symbol, Symbol], float],
        sensor_models,
        sensor_noises: dict[Symbol | tuple[Symbol, Symbol], float],
        calibration_map=None,
        *,
        config=None,
    ):
        """
        Provide an interface with required arguments to be more structured and.

        opinionated about how to Create this class. scikit-learn guides towards
        doing no construction or validation of the inputs in the __init__
        method, some is also done in this method with the goal of guiding the
        user earlier in the process.
        """
        parameters = {
            "symbolic_model": symbolic_model,
            "process_noise": process_noise,
            "sensor_models": sensor_models,
            "sensor_noises": sensor_noises,
            "calibration_map": calibration_map,
            "config": config,
        }

        estimator = cls(**parameters)
        return estimator

    def __init__(
        self,
        symbolic_model=None,
        process_noise=None,
        sensor_models=None,
        sensor_noises=None,
        calibration_map=None,
        *,
        config=None,
    ):
        self.symbolic_model = symbolic_model
        self.process_noise = process_noise
        self.sensor_models = sensor_models
        self.sensor_noises = sensor_noises
        self.calibration_map = calibration_map
        self.config = config

    def _flatten_process_noise(self, process_noise):
        for iIdx, iSymbol in enumerate(self.arglist_control):
            for jIdx, jSymbol in enumerate(self.arglist_control):
                if (iSymbol, jSymbol) in process_noise:
                    value = process_noise[(iSymbol, jSymbol)]
                elif (jSymbol, iSymbol) in process_noise:
                    value = process_noise[(jSymbol, iSymbol)]
                elif iSymbol == jSymbol and iSymbol in process_noise:
                    value = process_noise[iSymbol]
                else:
                    value = 0.0
                yield (iSymbol, jSymbol, value)

    def _sensor_noise_to_array(self, sensor_noises):
        matrix_sensor_noises = {}
        for key, model in self.sensor_models.items():
            assert isinstance(sensor_noises[key], dict)
            assert len(sensor_noises[key]) == len(self.sensor_models[key].keys())
            readings = sorted(list(model.keys()))
            ReadingCovariance = common.named_vector("ReadingCovariance", readings)

            matrix_sensor_noises[key] = ReadingCovariance.from_dict(sensor_noises[key])

        return matrix_sensor_noises

    def _compile_sensor_models(self, sensor_models):
        return {
            k: SensorModel(
                state_model=self.state_model,
                sensor_model=model,
                calibration_map=self.calibration_map,
                config=self.config,
            )
            for k, model in sensor_models.items()
        }

    def _flatten_dict_diagonal(self, mapping, arglist: list[Symbol]):
        for iIdx, iSymbol in enumerate(arglist):
            if (iSymbol, iSymbol) in mapping:
                value = mapping[(iSymbol, iSymbol)]
            elif iSymbol in mapping:
                value = mapping[iSymbol]
            else:
                value = 0.0
            yield value

    def _inverse_flatten_dict_diagonal(self, vector, arglist):
        for iIdx, iSymbol in enumerate(arglist):
            yield (iSymbol, vector[iIdx])

    def _flatten_scoring_params(self):
        """
        Note: Known limitation, this only flattens the diagonals to simplify.

        the `fit` optimizaiton problem
        """

        # Note: duplicated code from EKF
        arglist_control = sorted(
            list(self.symbolic_model.control), key=lambda x: x.name
        )

        flattened = list(
            self._flatten_dict_diagonal(self.process_noise, arglist_control)
        )

        for key, mapping in sorted(list(self.sensor_noises.items())):
            arglist = sorted(list(mapping.keys()))

            flattened.extend(self._flatten_dict_diagonal(mapping, arglist))

        return flattened

    def _inverse_flatten_scoring_params(self, flattened):
        # Note: duplicated code from EKF
        arglist_control = sorted(
            list(self.symbolic_model.control), key=lambda x: x.name
        )
        # Note: duplicated code from EKF
        control_size = len(self.symbolic_model.control)

        params = {k: getattr(self, k) for k in self.allowed_keys}
        controls, flattened = (
            flattened[:control_size],
            flattened[control_size:],
        )

        params["process_noise"] = dict(
            self._inverse_flatten_dict_diagonal(controls, arglist_control)
        )

        for key, mapping in sorted(list(self.sensor_noises.items())):
            sensor_size = len(mapping)
            sensor, flattened = flattened[:sensor_size], flattened[sensor_size:]

            arglist = sorted(list(mapping.keys()))

            params["sensor_noises"][key] = dict(
                self._inverse_flatten_dict_diagonal(sensor, arglist)
            )

        return params

    # Fit the model to data
    def fit(self, X, y=None, sample_weight=None) -> SklearnEKFAdapter:
        assert self.process_noise is not None
        assert self.sensor_models is not None
        assert self.sensor_noises is not None

        x0 = self._flatten_scoring_params()

        def minimize_this(x):
            holdout_params = dict(self.get_params())

            scoring_params = self._inverse_flatten_scoring_params(x)
            self.set_params(**scoring_params)

            score = self.score(X, y, sample_weight)

            self.set_params(**holdout_params)
            return score

        minimize_this(x0)

        result = minimize(minimize_this, x0, tol=1.0e-1)

        if not result.success:
            raise MinimizationFailure(result)

        soln_as_params = self._inverse_flatten_scoring_params(result.x)
        self.set_params(**soln_as_params)

        return self

    # Compute the squared Mahalanobis distances of given observations.
    def mahalanobis(self, X) -> NDArray:
        innovations, states, covariances = self.transform(X, include_states=True)
        if len(innovations.shape) == 1:
            innovations = np.reshape(innovations, (len(innovations), 1))
        n_samples, n_sensors = innovations.shape

        innovations = np.array(innovations).reshape((n_samples, n_sensors, 1))

        if np.any(innovations < 0.0):
            for idx, (x, innovation) in enumerate(
                zip(X.flatten(), innovations.flatten())
            ):
                if innovation < 0.0:
                    print(idx, x, innovation, states[idx])
            print("X")
            print(X.flatten())
            print("Innovations")
            print(innovations.flatten())
            raise AssertionError("innovations squared includes negative values")

        return innovations.flatten()

    # Compute something like the log-likelihood of X_test under the estimated Gaussian model.
    def score(
        self, X: Any, y=None, sample_weight=None, explain_score=False
    ) -> float | tuple[float, tuple[float, float, float, float, float, float]]:
        X = force_to_ndarray(X)
        y = force_to_ndarray(y)
        sample_weight = force_to_ndarray(sample_weight)

        mahalanobis_distance_squared = self.mahalanobis(X)
        normalized_innovations = np.sqrt(mahalanobis_distance_squared)

        if len(normalized_innovations) <= 0:
            raise ValueError(
                f"No innovations calculated from data shape {X.shape}. Calculated {normalized_innovations.shape}"
            )

        if sample_weight is None:
            avg = np.sum(np.square(np.mean(normalized_innovations)))
            var = np.sum(mahalanobis_distance_squared)
        else:
            avg = np.sum(np.square(np.mean(normalized_innovations * sample_weight)))
            var = np.sum(mahalanobis_distance_squared * sample_weight)

        # bias->0
        bias_weight = 1e1
        bias_score = avg

        if not np.isfinite(bias_score):
            raise ValueError(
                f"Bias Score not finite: {bias_score} from innovations {normalized_innovations}"
            )

        # variance->1
        # minima at var = 1, innovations match noise model
        variance_weight = 1e0
        variance_score = (1.0 / var + var) / 2.0

        if not np.isfinite(variance_score):
            raise ValueError(f"Variance Score not finite: {variance_score}")

        # prefer smaller matrix terms
        matrix_weight = 1e-2
        matrix_score = np.sum(
            np.square(
                list(
                    self._flatten_dict_diagonal(
                        self.process_noise, self.model_.arglist_control
                    )
                )
            )
        )
        for noise_mapping in self.sensor_noises.values():
            arglist = sorted(list(noise_mapping.keys()))
            matrix_score += np.sum(
                np.square(list(self._flatten_dict_diagonal(noise_mapping, arglist)))
            )

        result = (
            bias_weight * bias_score
            + variance_weight * variance_score
            + matrix_weight * matrix_score
        )
        if np.isnan(result):
            print(
                f"NaN result: {bias_weight} * {bias_score} + {variance_weight} * {variance_score} + {matrix_weight} * {matrix_score}"
            )

        if explain_score:
            return (
                result,
                (
                    bias_weight,
                    bias_score,
                    variance_weight,
                    variance_score,
                    matrix_weight,
                    matrix_score,
                ),
            )

        return result

    # Transform readings to innovations
    def transform(
        self, X: Any, include_states=False
    ) -> NDArray | tuple[NDArray, NDArray, NDArray]:
        assert self.symbolic_model is not None
        assert self.process_noise is not None
        self.model_ = compile_ekf(
            self.symbolic_model,
            self.process_noise,
            self.sensor_models,
            self.sensor_noises,
            self.calibration_map,
            config=self.config,
        )
        if len(self.model_.sensor_models) <= 0:
            raise ValueError("Sensor Models required to calculate innovation")

        X = force_to_ndarray(X)
        if len(X.shape) == 1:
            X = np.reshape(X, (len(X), 1))

        n_samples, n_features = X.shape

        dt = 0.1

        state = self.model_.State()
        covariance = self.model_.Covariance()

        innovations = []
        states = [state]
        covariances = [covariance]

        for key in sorted(list(self.model_.sensor_models)):
            sensor_size = len(self.model_.sensor_models[key])
            # Incomplete thought

        for idx in range(X.shape[0]):
            controls_input, the_rest = (
                X[idx, : self.model_.control_size],
                X[idx, self.model_.control_size :],
            )
            controls_input = self.model_.Control.from_data(
                controls_input.reshape((self.model_.control_size, 1))
            )

            state, covariance = self.model_.process_model(
                dt, state, covariance, controls_input
            )

            innovation = []

            for key in sorted(list(self.model_.sensor_models)):
                sensor_size = len(self.model_.sensor_models[key])

                sensor_input, the_rest = (
                    the_rest[:sensor_size],
                    the_rest[sensor_size:],
                )
                sensor_input = self.model_.make_reading(
                    key, data=sensor_input.reshape((sensor_size, 1))
                )

                state, covariance = self.model_.sensor_model(
                    state=state,
                    covariance=covariance,
                    sensor_key=key,
                    sensor_reading=sensor_input,
                )
                # Normalized by the uncertainty at the time of the measurement
                # Mahalanobis distance = sqrt((x - u).T * S^{-1} * (x - u))
                # for:
                #   u: predicted sensor readings
                #   x: sensor readings
                #   S: predicted sensor variance
                innovation.append(
                    float(
                        np.matmul(
                            np.matmul(
                                self.model_.innovations[key].T,
                                np.linalg.inv(
                                    self.model_.sensor_prediction_uncertainty[key]
                                ),
                            ),
                            self.model_.innovations[key],
                        )
                    )
                )

            states.append(state)
            covariances.append(covariance)
            innovations.append(innovation)
            assert innovations[-1] is not None

        # minima at x = 1, innovations match noise model
        if include_states:
            return (
                np.array(innovations, dtype="float"),
                np.array(states),
                np.array(covariances),
            )

        return np.array(innovations, dtype="float")

    # Fit the model to data and transform readings to innovations
    def fit_transform(self, X, y=None) -> NDArray | tuple[NDArray, NDArray, NDArray]:
        # TODO(buck): Implement the combined version (return innovations calculated while fitting)
        self.fit(X, y)
        return self.transform(X)

    # Get parameters for this estimator.
    def get_params(self, deep=True) -> dict[str, Any]:
        return {
            "symbolic_model": self.symbolic_model,
            "process_noise": self.process_noise,
            "sensor_models": self.sensor_models,
            "sensor_noises": self.sensor_noises,
            "calibration_map": self.calibration_map,
            "config": self.config,
        }

    # Set the parameters of this estimator.
    def set_params(self, **params) -> SklearnEKFAdapter:
        for key in params:
            if key in self.allowed_keys:
                setattr(self, key, params[key])
            elif key in dataclasses.asdict(self.config):
                mutable_version = dataclasses.asdict(self.config)
                mutable_version[key] = params[key]
                self.config = Config(**mutable_version)
            else:
                raise ModelConstructionError(
                    f"set_params called with invalid key {key}"
                )

        return self

    def export_python(self):
        return compile_ekf(
            self.symbolic_model,
            self.process_noise,
            self.sensor_models,
            self.sensor_noises,
            self.calibration_map,
            config=self.config,
        )
