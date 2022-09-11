import numpy as np

from sympy import Matrix
from sympy.utilities.lambdify import lambdify
from numba import njit

MODULES = ["scipy", "numpy", "math"]


class Model(object):
    """Python implementation of the model"""

    def __init__(self, symbolic_model, config):
        # assert isinstance(symbolic_model, ui.Model)
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)

        self.arglist = (
            [symbolic_model.dt]
            + sorted(list(symbolic_model.state), key=lambda x: x.name)
            + sorted(list(symbolic_model.control), key=lambda x: x.name)
        )

        # TODO(buck): Common Subexpression Elimination supports multiple inputs, so use common subexpression elimination across state calculations
        self._impl = [
            lambdify(
                self.arglist,
                symbolic_model.state_model[a],
                modules=config.python_modules,
                cse=config.common_subexpression_elimination,
            )
            for a in self.arglist[1 : 1 + len(symbolic_model.state)]
        ]

        if config.compile:
            self._impl = [njit(i) for i in self._impl]

    def _model(self, dt, state, control):
        assert isinstance(dt, float)
        assert isinstance(state, np.ndarray)
        assert isinstance(control, np.ndarray)

        for impl in self._impl:
            yield impl(dt, *state, *control)

    # TODO(buck): numpy -> numpy if not compiled
    #   - Given the arglist, refactor expressions to work with state vectors
    #   - Transparently convert states, controls to numpy so that it's always numpy -> numpy
    def model(self, dt, state, control=None):
        if control is None:
            control = []
        next_state = np.zeros(state.shape)
        for i, val in enumerate(self._model(dt, state, control)):
            next_state[i, 0] = val
        return next_state


class ExtendedKalmanFilter(object):
    def __init__(
        self, state_model, process_noise, sensor_models, sensor_noises, config
    ):
        assert isinstance(config, Config)

        self.state_size = len(state_model.state)
        self.control_size = len(state_model.control)

        self._construct_process(state_model, process_noise, config)
        self._construct_sensors(state_model, sensor_models, sensor_noises, config)

    def _construct_process(self, state_model, process_noise, config):
        self.state_model = Model(state_model, config)
        self.process_noise = process_noise

        process_matrix = Matrix(
            [
                state_model.state_model[a]
                for a in self.state_model.arglist[1 : 1 + self.state_size]
            ]
        )
        symbolic_process_jacobian = process_matrix.jacobian(
            self.state_model.arglist[1 : 1 + self.state_size]
        )
        # TODO(buck): This assertion won't necessarily hold if CSE is on across states
        assert symbolic_process_jacobian.shape == (
            self.state_size,
            self.state_size,
        )

        symbolic_control_jacobian = process_matrix.jacobian(
            self.state_model.arglist[-self.control_size :]
        )

        self._impl_process_jacobian = [
            lambdify(
                self.state_model.arglist,
                expr,
                modules=config.python_modules,
                cse=config.common_subexpression_elimination,
            )
            for expr in symbolic_process_jacobian
        ]
        assert len(self._impl_process_jacobian) == self.state_size ** 2
        # TODO(buck): parameterized tests with compile=False and compile=True. Generically, parameterize tests over all config (or a useful subset of all configs)
        if config.compile:
            self._impl_process_jacobian = [njit(i) for i in self._impl_process_jacobian]

        self._impl_control_jacobian = [
            lambdify(
                self.state_model.arglist,
                expr,
                modules=config.python_modules,
                cse=config.common_subexpression_elimination,
            )
            for expr in symbolic_control_jacobian
        ]
        assert len(self._impl_control_jacobian) == self.control_size * self.state_size
        if config.compile:
            self._impl_control_jacobian = [njit(i) for i in self._impl_control_jacobian]

    def _construct_sensors(self, state_model, sensor_models, sensor_noises, config):
        assert sorted(list(sensor_models.keys())) == sorted(list(sensor_noises.keys()))
        self.sensor_models = {
            k: StateModel(model, config) for k, model in sensor_models
        }
        self.sensor_noises = sensor_noises

        self.arglist_sensor = sorted(list(state_model.state), key=lambda x: x.name)

        self._impl_sensor_jacobians = {}

        for k, sensor_model in self.sensor_models:
            sensor_size = len(sensor_model.readings)

            sensor_matrix = Matrix(
                [sensor_model.sensor_model[r] for r in sensor_model.readings]
            )
            symbolic_sensor_jacobian = sensor_matrix.jacobian(self.arglist_sensor)
            # TODO(buck): This assertion won't necessarily hold if CSE is on across states
            assert symbolic_sensor_jacobian.shape == (
                sensor_size,
                self.state_size,
            )

            impl_sensor_jacobian = [
                lambdify(
                    self.arglist_sensor,
                    expr,
                    modules=config.python_modules,
                    cse=config.common_subexpression_elimination,
                )
                for expr in symbolic_sensor_jacobian
            ]
            assert len(impl_sensor_jacobian) == sensor_size * self.state_size
            # TODO(buck): warm jit by calling the compiled functions with a zero state vector
            # TODO(buck): allow for compiling only process, sensors or list of specific sensors
            if config.compile:
                impl_sensor_jacobian = [njit(i) for i in impl_sensor_jacobian]

            self._impl_sensor_jacobians[k] = impl_sensor_jacobian

    def process_jacobian(self, dt, state, control):
        jacobian = np.zeros((self.state_size, self.state_size))
        # TODO(buck) skip known zeros / sparse bits of the matrix
        # TODO(buck) Could you precompute all constant parts of the matrix? (e.g. =1.0)
        for row in range(self.state_size):
            for col in range(self.state_size):
                jacobian[row, col] = self._impl_process_jacobian[
                    row * self.state_size + col
                ](dt, *state, *control)
        return jacobian

    def control_jacobian(self, dt, state, control):
        jacobian = np.zeros((self.state_size, self.control_size))
        # TODO(buck) skip known zeros / sparse bits of the matrix
        # TODO(buck) Could you precompute all constant parts of the matrix? (e.g. =1.0)
        for row in range(self.state_size):
            for col in range(self.control_size):
                jacobian[row, col] = self._impl_control_jacobian[
                    row * self.control_size + col
                ](dt, *state, *control)
        return jacobian

    def process_model(self, dt, state, covariance, control):
        try:
            assert isinstance(state, np.ndarray)
            assert isinstance(covariance, np.ndarray)
            assert isinstance(control, np.ndarray)
        except AssertionError:
            print(
                "process_model(dt: %s, state: %s, covariance: %s, control: %s)"
                % (type(dt), type(state), type(covariance), type(control))
            )
            raise

        try:
            assert state.shape == (self.state_size, 1)
            assert covariance.shape == (self.state_size, self.state_size)
            assert control.shape == (self.control_size, 1)
        except AssertionError:
            print(
                "process_model(dt: %s, state: %s, covariance: %s, control: %s)"
                % (dt, state.shape, covariance.shape, control.shape)
            )

        # TODO(buck): CSE across the whole process computation (model, jacobians)
        G_t = self.process_jacobian(dt, state, control)
        V_t = self.control_jacobian(dt, state, control)

        next_state_covariance = np.matmul(G_t, np.matmul(covariance, G_t.transpose()))
        assert next_state_covariance.shape == covariance.shape
        next_control_covariance = np.matmul(
            V_t, np.matmul(self.process_noise, V_t.transpose())
        )
        assert next_control_covariance.shape == covariance.shape
        next_covariance = next_state_covariance + next_control_covariance
        assert next_covariance.shape == covariance.shape

        next_state = self.state_model.model(dt, state, control)
        return next_state, next_covariance

    def sensor_model(self, state, covariance, sensor_key, sensor_reading):
        model_impl = self.sensor_models[sensor_key]
        sensor_size = len(model_impl.readings)
        Q_t = model_noise = self.sensor_noises[sensor_key]

        try:
            assert isinstance(state, np.ndarray)
            assert isinstance(covariance, np.ndarray)
            assert isinstance(sensor_reading, np.ndarray)
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

        try:
            assert state.shape == (self.state_size, 1)
            assert covariance.shape == (sensor_size, self.state_size)
            assert sensor_reading.shape == (sensor_size, 1)
        except AssertionError:
            print(
                "sensor_model(state: %s, covariance: %s, sensor_key, sensor_reading: %s)"
                % (dt, state.shape, covariance.shape, sensor_reading.shape)
            )

        # TODO(buck): Assert model noise is the correct shape at construction time
        expected_reading = model_impl.model(state)

        H_t = self.sensor_jacobian(sensor_key, state)

        S_t = sensor_prediction_uncertainty = (
            np.matmul(H_t, np.matmul(covariance, H_t.transpose())) + Q_t
        )
        S_inv = np.linalg.inv(S_t)

        K_t = kalman_gain = np.matmul(covariance, np.matmul(H_t.transpose(), S_inv))

        innovation = sensor_reading - expected_reading

        next_state = state + np.matmul(K_t, innovation)

        next_covariance = covariance - np.matmul(K_t, np.matmul(H_t, covariance))

        return next_state, next_covariance


class Config(object):
    def __init__(
        self,
        compile=False,
        common_subexpression_elimination=True,
        python_modules=tuple(MODULES),
    ):
        self.compile = compile
        self.common_subexpression_elimination = common_subexpression_elimination
        self.python_modules = python_modules


def compile(symbolic_model, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    return Model(symbolic_model, config)


def compile_ekf(
    state_model, process_noise, sensor_models, sensor_noises, *, config=None
):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    return ExtendedKalmanFilter(
        state_model, process_noise, sensor_models, sensor_noises, config
    )
