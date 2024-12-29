from __future__ import annotations

from collections import namedtuple
from math import sqrt

import numpy as np
import sympy
from numpy.typing import NDArray
from sympy import Matrix, Symbol

from formak import common
from formak.compiler.config import Config
from formak.compiler.basic_block import BasicBlock
from formak.runtime.model import Model
from formak.runtime.sensor_model import SensorModel
from formak.runtime.assertions import assert_valid_covariance


StateAndCovariance = namedtuple("StateAndCovariance", ["state", "covariance"])


class ExtendedKalmanFilter:
    def __init__(
        self,
        state_model,
        process_noise: dict[Symbol | tuple[Symbol, Symbol], float],
        sensor_models: dict[str, sympy.core.expr.Expr],
        sensor_noises: dict[str, dict[Symbol | tuple[Symbol, Symbol], float]],
        config: Config,
        calibration_map: dict[Symbol, float] | None = None,
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

    def _construct_process(
        self,
        state_model,
        process_noise: dict[Symbol | tuple[Symbol, Symbol], float],
        calibration_map: dict[Symbol, float],
        config: Config,
    ) -> None:
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
        assert_valid_covariance(self.process_noise)

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
        self,
        state_model: common.UiModelBase,
        sensor_models: dict[str, sympy.core.expr.Expr],
        sensor_noises: dict[str, dict[Symbol | tuple[Symbol, Symbol], float]],
        calibration_map: dict[Symbol, float],
        config: Config,
    ) -> None:
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
        assert_valid_covariance(covariance.data)
        assert_valid_covariance(self.process_noise)

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
        assert_valid_covariance(next_state_covariance)

        next_control_covariance = np.matmul(
            V_t, np.matmul(self.process_noise, V_t.transpose())
        )
        assert next_control_covariance.shape == covariance.shape
        assert_valid_covariance(next_control_covariance)

        next_covariance = next_state_covariance + next_control_covariance
        assert next_covariance.shape == covariance.shape
        assert_valid_covariance(next_covariance)

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
        assert_valid_covariance(covariance.data)

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

        assert_valid_covariance(covariance.data)

        self.sensor_prediction_uncertainty[sensor_key] = S_t = (
            np.matmul(H_t, np.matmul(covariance.data, H_t.transpose())) + Q_t.data
        )
        assert_valid_covariance(S_t, name="Sensor Uncertainty")

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
