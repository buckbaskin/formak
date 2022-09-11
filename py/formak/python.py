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
                modules=MODULES,
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
    def __init__(self, state_model, sensor_models, config):
        assert isinstance(config, Config)

        self.state_size = len(state_model.state)
        self.control_size = len(state_model.control)

        self.state_model = Model(state_model, config)
        self.sensor_models = sensor_models

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
                modules=MODULES,
                cse=config.common_subexpression_elimination,
            )
            for expr in symbolic_process_jacobian
        ]
        assert len(self._impl_process_jacobian) == self.state_size ** 2
        if config.compile:
            self._impl_process_jacobian = [njit(i) for i in self._impl_process_jacobian]

        self._impl_control_jacobian = [
            lambdify(
                self.state_model.arglist,
                expr,
                modules=MODULES,
                cse=config.common_subexpression_elimination,
            )
            for expr in symbolic_control_jacobian
        ]
        assert len(self._impl_control_jacobian) == self.control_size * self.state_size
        if config.compile:
            self._impl_control_jacobian = [njit(i) for i in self._impl_control_jacobian]

        # TODO(buck): allow configurable process noise
        self.process_noise = np.eye(self.control_size)

        self.arglist_sensor = sorted(
            list(state_model.state), key=lambda x: x.name
        ) + sorted(list(state_model.control), key=lambda x: x.name)

    def process_jacobian(self, dt, state, control):
        jacobian = np.zeros((self.state_size, self.state_size))
        # TODO(buck) skip known zeros / sparse bits of the matrix
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

    def sensor_model(self, state, covariance, sensor):
        # TODO(buck): Implement EKF sensor update
        return state, covariance


class Config(object):
    def __init__(self, compile=False, common_subexpression_elimination=True):
        self.compile = compile
        self.common_subexpression_elimination = common_subexpression_elimination
        # TODO(buck): introduce config for modules, custom sin implementation, etc.


def compile(symbolic_model, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    return Model(symbolic_model, config)


def compile_ekf(state_model, sensor_models, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    return ExtendedKalmanFilter(state_model, sensor_models, config)
