import numpy as np

from scipy.optimize import minimize
from sympy import Matrix
from sympy.utilities.lambdify import lambdify
from numba import njit

DEFAULT_MODULES = ("scipy", "numpy", "math")


class Model(object):
    """Python implementation of the model"""

    def __init__(self, symbolic_model, config):
        # TODO(buck): Enable mypy for type checking
        # TODO(buck): Move all type assertions to either __init__ (constructor) or mypy?
        # assert isinstance(symbolic_model, ui.Model)
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)

        self.state_size = len(symbolic_model.state)
        self.control_size = len(symbolic_model.control)

        self.arglist_state = sorted(list(symbolic_model.state), key=lambda x: x.name)
        self.arglist = (
            [symbolic_model.dt]
            + self.arglist_state
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
            for a in self.arglist_state
        ]

        if config.compile:
            self._impl = [njit(i) for i in self._impl]

            if config.warm_jit:
                # Pre-warm Jit by calling with known values/structure
                default_dt = 0.1
                default_state = np.zeros((self.state_size, 1))
                default_control = np.zeros((self.control_size, 1))
                for jit_impl in self._impl:
                    for i in range(5):
                        jit_impl(default_dt, *default_state, *default_control)

    # TODO(buck): numpy -> numpy if not compiled
    #   - Given the arglist, refactor expressions to work with state vectors
    #   - Transparently convert states, controls to numpy so that it's always numpy -> numpy
    def model(self, dt, state, control_vector=None):
        if control_vector is None:
            control_vector = np.zeros((0, 0))

        assert isinstance(dt, float)
        assert isinstance(state, np.ndarray)
        assert isinstance(control_vector, np.ndarray)

        assert state.shape == (self.state_size, 1)
        assert control_vector.shape == (self.control_size, min(self.control_size, 1))

        next_state = np.zeros((self.state_size, 1))
        for i, impl in enumerate(self._impl):
            next_state[i, 0] = impl(dt, *state, *control_vector)
        return next_state


class SensorModel(object):
    def __init__(self, state_model, sensor_model, config):
        self.readings = sorted(list(sensor_model.keys()))
        self.sensor_models = sensor_model

        self.sensor_size = len(self.readings)
        self.state_size = len(state_model.state)

        self.arglist = sorted(list(state_model.state), key=lambda x: x.name)

        self._impl = [
            lambdify(
                self.arglist,
                sensor_model[k],
                modules=config.python_modules,
                cse=config.common_subexpression_elimination,
            )
            for k in self.readings
        ]

    def __len__(self):
        return len(self.sensor_models)

    def model(self, state):
        assert isinstance(state, np.ndarray)
        assert state.shape == (self.state_size, 1)

        reading = np.zeros((self.sensor_size, 1))
        for i, impl in enumerate(self._impl):
            reading[i, 0] = impl(*state)
        return reading


class ExtendedKalmanFilter(object):
    def __init__(
        self, state_model, process_noise, sensor_models, sensor_noises, config
    ):
        assert isinstance(config, Config)

        self.state_size = len(state_model.state)
        self.control_size = len(state_model.control)
        self.params = {
            "process_noise": process_noise,
            "sensor_models": sensor_models,
            "sensor_noises": sensor_noises,
            "compile": compile,
        }

        self._construct_process(state_model, process_noise, config)
        self._construct_sensors(state_model, sensor_models, sensor_noises, config)

    def _construct_process(self, state_model, process_noise, config):
        self.state_model = Model(state_model, config)
        self.params["process_noise"] = process_noise

        # TODO(buck): Reorder state vector (arglist*) to take advantage of sparse blocks (e.g. assign in a block, skip a block, etc)

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
        assert len(self._impl_process_jacobian) == self.state_size**2

        # TODO(buck): parameterized tests with compile=False and compile=True. Generically, parameterize tests over all config (or a useful subset of all configs)
        if config.compile:
            self._impl_process_jacobian = [njit(i) for i in self._impl_process_jacobian]

            if config.warm_jit:
                # Pre-warm Jit by calling with known values/structure
                default_dt = 0.1
                default_state = np.zeros((self.state_size, 1))
                default_control = np.zeros((self.control_size, 1))
                for jit_impl in self._impl_process_jacobian:
                    for i in range(5):
                        jit_impl(default_dt, *default_state, *default_control)

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

            if config.warm_jit:
                # Pre-warm Jit by calling with known values/structure
                default_dt = 0.1
                default_state = np.zeros((self.state_size, 1))
                default_control = np.zeros((self.control_size, 1))
                for jit_impl in self._impl_control_jacobian:
                    for i in range(5):
                        jit_impl(default_dt, *default_state, *default_control)

    def _construct_sensors(self, state_model, sensor_models, sensor_noises, config):
        assert sorted(list(sensor_models.keys())) == sorted(list(sensor_noises.keys()))

        self.params["sensor_models"] = {
            k: SensorModel(state_model, model, config)
            for k, model in sensor_models.items()
        }
        for k in self.params["sensor_models"].keys():
            assert sensor_noises[k].shape == (
                self.params["sensor_models"][k].sensor_size,
                self.params["sensor_models"][k].sensor_size,
            )

        self.params["sensor_noises"] = sensor_noises

        self.arglist_sensor = sorted(list(state_model.state), key=lambda x: x.name)

        self._impl_sensor_jacobians = {}

        for k, sensor_model in self.params["sensor_models"].items():
            sensor_size = len(sensor_model.readings)

            sensor_matrix = Matrix(
                [sensor_model.sensor_models[r] for r in sensor_model.readings]
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

            # TODO(buck): allow for compiling only process, sensors or list of specific sensors
            if config.compile:
                impl_sensor_jacobian = [njit(i) for i in impl_sensor_jacobian]

                if config.warm_jit:
                    # Pre-warm Jit by calling with known values/structure
                    default_state = np.zeros((self.state_size, 1))
                    for jit_impl in impl_sensor_jacobian:
                        for i in range(5):
                            jit_impl(*default_state)

            self._impl_sensor_jacobians[k] = impl_sensor_jacobian

        self.innovations = {}
        self.sensor_prediction_uncertainty = {}

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

    def sensor_jacobian(self, sensor_key, state):
        sensor_size = self.params["sensor_models"][sensor_key].sensor_size
        jacobian = np.zeros((sensor_size, self.state_size))

        impl_sensor_jacobian = self._impl_sensor_jacobians[sensor_key]

        for row in range(sensor_size):
            for col in range(self.state_size):
                jacobian[row, col] = impl_sensor_jacobian[row * sensor_size + col](
                    *state
                )
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
            raise

        # TODO(buck): CSE across the whole process computation (model, jacobians)
        G_t = self.process_jacobian(dt, state, control)
        V_t = self.control_jacobian(dt, state, control)

        next_state_covariance = np.matmul(G_t, np.matmul(covariance, G_t.transpose()))
        assert next_state_covariance.shape == covariance.shape

        next_control_covariance = np.matmul(
            V_t, np.matmul(self.params["process_noise"], V_t.transpose())
        )
        assert next_control_covariance.shape == covariance.shape

        next_covariance = next_state_covariance + next_control_covariance
        assert next_covariance.shape == covariance.shape

        next_state = self.state_model.model(dt, state, control)
        assert next_state.shape == state.shape

        return next_state, next_covariance

    def sensor_model(self, sensor_key, state, covariance, sensor_reading):
        model_impl = self.params["sensor_models"][sensor_key]
        sensor_size = len(model_impl.readings)
        Q_t = model_noise = self.params["sensor_noises"][sensor_key]

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
            assert covariance.shape == (self.state_size, self.state_size)
            assert sensor_reading.shape == (sensor_size, 1)
            assert Q_t.shape == (sensor_size, sensor_size)
        except AssertionError:
            print(
                "sensor_model(state: %s, covariance: %s, sensor_key, sensor_reading: %s)"
                % (state.shape, covariance.shape, sensor_reading.shape)
            )
            raise

        expected_reading = model_impl.model(state)

        H_t = self.sensor_jacobian(sensor_key, state)
        assert H_t.shape == (sensor_size, self.state_size)

        self.sensor_prediction_uncertainty[sensor_key] = S_t = (
            np.matmul(H_t, np.matmul(covariance, H_t.transpose())) + Q_t
        )
        S_inv = np.linalg.inv(S_t)

        K_t = kalman_gain = np.matmul(covariance, np.matmul(H_t.transpose(), S_inv))

        self.innovations[sensor_key] = innovation = sensor_reading - expected_reading
        # TODO(buck): is innovation normalized by variance?

        next_covariance = covariance - np.matmul(K_t, np.matmul(H_t, covariance))
        assert next_covariance.shape == covariance.shape

        next_state = state + np.matmul(K_t, innovation)
        assert next_state.shape == state.shape

        return next_state, next_covariance

    ### scikit-learn / sklearn interface ###

    def _flatten_scoring_params(self, params):
        # "process_noise": np.eye(1),
        # "sensor_models": {"simple": {ui.Symbol("v"): ui.Symbol("v")}},
        # "sensor_noises": {"simple": np.eye(1)},
        flattened = list(np.diagonal(params["process_noise"]))
        for key in sorted(list(params["sensor_models"])):
            flattened.extend(np.diagonal(params["sensor_noises"][key]))

        return flattened

    def _inverse_flatten_scoring_params(self, flattened):
        params = {k: v for k, v in self.params.items()}

        controls, flattened = (
            flattened[: self.control_size],
            flattened[self.control_size :],
        )

        np.fill_diagonal(params["process_noise"], controls)

        for key in sorted(list(self.params["sensor_models"])):
            sensor_size = len(np.diagonal(params["sensor_noises"][key]))
            sensor, flattened = flattened[:sensor_size], flattened[sensor_size:]
            np.fill_diagonal(params["sensor_noises"][key], sensor)

        return params

    # Fit the model to data
    def fit(self, X, y=None, sample_weight=None):
        # TODO(buck): Figure out dt, add it as the first element of X
        dt = 0.1

        assert self.params["process_noise"] is not None
        assert self.params["sensor_models"] is not None
        assert self.params["sensor_noises"] is not None

        x0 = self._flatten_scoring_params(self.params)
        # TODO(buck): implement parameter fitting, y ignored
        def minimize_this(x):
            holdout_params = dict(self.get_params())

            scoring_params = self._inverse_flatten_scoring_params(x)
            self.set_params(**scoring_params)

            score = self.score(X, y, sample_weight)

            self.set_params(**holdout_params)
            return score

        minimize_this(x0)

        result = minimize(minimize_this, x0)

        if not result.success:
            print("success", result.success, result.message)
            assert result.success

        soln_as_params = self._inverse_flatten_scoring_params(result.x)
        self.set_params(**soln_as_params)

        return self

    # Compute the squared Mahalanobis distances of given observations.
    def mahalanobis(self, X):
        # TODO(buck): mahalanobis
        n_samples, n_features = X.shape
        shape = (n_samples,)
        return np.zeros(shape)

    # Compute the log-likelihood of X_test under the estimated Gaussian model.
    def score(self, X, y=None, sample_weight=None):
        dt = 0.1

        state = np.zeros((self.state_size, 1))
        covariance = np.eye(self.state_size)

        innovations = []

        for idx in range(X.shape[0]):
            controls_input, the_rest = (
                X[idx, : self.control_size],
                X[idx, self.control_size :],
            )
            controls_input = controls_input.reshape((self.control_size, 1))

            state, covariance = self.process_model(
                dt, state, covariance, controls_input
            )

            innovation = []

            for key in sorted(list(self.params["sensor_models"])):
                sensor_size = len(self.params["sensor_models"][key])

                sensor_input, the_rest = (
                    the_rest[:sensor_size],
                    the_rest[sensor_size:],
                )
                sensor_input = sensor_input.reshape((sensor_size, 1))

                state, covariance = self.sensor_model(
                    key, state, covariance, sensor_input
                )
                # Normalized by the uncertainty at the time of the measurement
                innovation.append(
                    float(
                        np.matmul(
                            self.innovations[key],
                            np.linalg.inv(self.sensor_prediction_uncertainty[key]),
                        )
                    )
                )

            innovations.append(innovation)

        x = np.sum(np.square(innovations))

        # minima at x = 1, innovations match noise model
        return (1.0 / x + x) / 2.0

    # Transform readings to innovations
    def transform(self, X):
        # TODO(buck): transform
        n_samples, n_features = X.shape
        output_features = n_features - self.control_size

        dt = 0.1

        state = np.zeros((self.state_size, 1))
        covariance = np.eye(self.state_size)

        innovations = []

        for idx in range(X.shape[0]):
            controls_input, the_rest = (
                X[idx, : self.control_size],
                X[idx, self.control_size :],
            )
            controls_input = controls_input.reshape((self.control_size, 1))

            state, covariance = self.process_model(
                dt, state, covariance, controls_input
            )

            innovation = []

            for key in sorted(list(self.params["sensor_models"])):
                sensor_size = len(self.params["sensor_models"][key])

                sensor_input, the_rest = (
                    the_rest[:sensor_size],
                    the_rest[sensor_size:],
                )
                sensor_input = sensor_input.reshape((sensor_size, 1))

                state, covariance = self.sensor_model(
                    key, state, covariance, sensor_input
                )
                # Normalized by the uncertainty at the time of the measurement
                innovation.append(
                    float(
                        np.matmul(
                            self.innovations[key],
                            np.linalg.inv(self.sensor_prediction_uncertainty[key]),
                        )
                    )
                )

            innovations.append(innovation)
            assert innovations[-1] is not None

        # minima at x = 1, innovations match noise model
        print("innovations")
        print(innovations)
        return np.array(innovations)

    # Fit the model to data and transform readings to innovations
    def fit_transform(self, X, y=None):
        # TODO(buck): Implement the combined version (return innovations calculated while fitting)
        self.fit(X, y)
        return self.transform(X)

    # Get parameters for this estimator.
    def get_params(self, deep=True) -> dict:
        return self.params

    # Set the parameters of this estimator.
    def set_params(self, **params):
        for p in params:
            if p in self.params:
                self.params[p] = params[p]

        return self


class Config(object):
    def __init__(
        self,
        compile=False,
        warm_jit=None,
        common_subexpression_elimination=True,
        python_modules=DEFAULT_MODULES,
    ):
        if warm_jit is None:
            warm_jit = compile
        self.compile = compile
        self.warm_jit = warm_jit
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
