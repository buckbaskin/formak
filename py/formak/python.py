from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Dict

import numpy as np
from formak.exceptions import MinimizationFailure, ModelConstructionError
from numba import njit
from scipy.optimize import minimize
from sympy import Matrix, Symbol, cse
from sympy.utilities.lambdify import lambdify

from formak import common

DEFAULT_MODULES = ("scipy", "numpy", "math")


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

        replacements = []
        reduced_exprs = [symbolic_model.state_model[a] for a in self.arglist_state]

        if config.common_subexpression_elimination:
            replacements, reduced_exprs = cse(
                reduced_exprs, symbols=(Symbol(f"_t{i}") for i in count())
            )

        temporaries = [r[0] for r in replacements]
        self._impl_prefix = []
        for i in range(len(replacements)):
            self._impl_prefix.append(
                (
                    temporaries[i],
                    lambdify(
                        self.arglist + temporaries[:i],
                        replacements[i][1],
                        modules=config.python_modules,
                        cse=False,
                    ),
                )
            )

        self._impl = [
            lambdify(
                self.arglist + temporaries,
                symbolic_model.state_model[a],
                modules=config.python_modules,
                cse=False,
            )
            for a in self.arglist_state
        ]

        if config.compile:
            # TODO(buck): Compile _impl_prefix as well
            self._impl = [njit(i) for i in self._impl]

            if config.warm_jit:
                # Pre-warm Jit by calling with known values/structure
                default_dt = 0.1
                default_state = np.zeros((self.state_size, 1))
                default_control = np.zeros((self.control_size, 1))
                default_calibration = np.zeros((self.calibration_size, 1))
                default_temporaries = np.zeros((len(temporaries), 1))
                for jit_impl in self._impl:
                    for _i in range(5):
                        jit_impl(
                            default_dt,
                            *default_state,
                            *default_calibration,
                            *default_control,
                            *default_temporaries,
                        )

    def model(self, dt, state, control_vector=None):
        if control_vector is None:
            if self.control_size > 0:
                raise TypeError(
                    "model() missing 1 required positional argument: 'control_vector'"
                )
            control_vector = np.zeros((0, 1))

        assert isinstance(dt, float)
        assert isinstance(state, np.ndarray)
        assert isinstance(control_vector, np.ndarray)

        assert state.shape == (self.state_size, 1)
        assert control_vector.shape == (self.control_size, 1)

        start_temps = datetime.now()
        temporaries = {}
        for target, expr in self._impl_prefix:
            temporaries[str(target)] = expr(
                dt, *state, *self.calibration_vector, *control_vector, **temporaries
            )

        end_temps = datetime.now()

        next_state = np.zeros((self.state_size, 1))
        for i, (state_id, impl) in enumerate(zip(self.arglist_state, self._impl)):
            try:
                result = impl(
                    dt, *state, *self.calibration_vector, *control_vector, **temporaries
                )
                next_state[i, 0] = result
            except TypeError:
                print(
                    "TypeError when trying to process process model for state %s"
                    % (state_id,)
                )
                print(f"given: {state}, {self.calibration_vector}, {control_vector}")
                print("expected: float")
                if "result" in locals():
                    print("found: {}, {}".format(type(result), result))
                raise

        end_impl = datetime.now()

        print(f"Timing impl {end_impl - end_temps} temps {end_temps - start_temps}")

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

        self.calibration_vector = np.array(
            [[calibration_map[k] for k in self.arglist_calibration]]
        ).transpose()
        if self.calibration_vector.shape != (self.calibration_size, 1):
            raise ModelConstructionError(
                f"calibration vector shape {self.calibration_vector.shape} doesn't match expected shape {(self.calibration_size, 1)}"
            )

        self._impl = [
            lambdify(
                self.arglist,
                sensor_model[k],
                modules=config.python_modules,
                cse=config.common_subexpression_elimination,
            )
            for k in self.readings
        ]

        ## "Pre-flight" Checks

        # Pre-check model for type errors
        self.model(np.zeros((self.state_size, 1)))

    def __len__(self):
        return len(self.sensor_models)

    def model(self, state_vector):
        assert isinstance(state_vector, np.ndarray)
        assert state_vector.shape == (self.state_size, 1)

        reading = np.zeros((self.sensor_size, 1))
        for i, (reading_id, impl) in enumerate(zip(self.readings, self._impl)):
            try:
                result = impl(*state_vector, *self.calibration_vector)
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
        return reading


StateAndCovariance = namedtuple("StateAndCovariance", ["state", "covariance"])


class ExtendedKalmanFilter:
    def __init__(
        self,
        state_model,
        process_noise: Dict[Symbol, float],
        sensor_models,
        sensor_noises,
        config,
        calibration_map=None,
    ):
        if calibration_map is None:
            calibration_map = {}

        assert isinstance(config, Config)
        assert isinstance(process_noise, dict)
        assert isinstance(calibration_map, dict)

        self.state_size = len(state_model.state)
        self.control_size = len(state_model.control)
        self.calibration_size = len(state_model.calibration)

        self.params = {
            "process_noise": None,
            "sensor_models": sensor_models,
            "sensor_noises": sensor_noises,
            "calibration_map": calibration_map,
            "compile": compile,
        }

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

        for iIdx, iSymbol in enumerate(self._state_model.arglist_control):
            for jIdx, jSymbol in enumerate(self._state_model.arglist_control):
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

        self.params["process_noise"] = process_noise_matrix

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
            symbolic_control_jacobian = process_matrix.jacobian(
                self._state_model.arglist_control
            )

        self._impl_process_jacobian = [
            lambdify(
                self._state_model.arglist,
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
                default_calibration = np.zeros((self.calibration_size, 1))
                default_control = np.zeros((self.control_size, 1))
                for jit_impl in self._impl_process_jacobian:
                    for _i in range(5):
                        jit_impl(
                            default_dt,
                            *default_state,
                            *default_calibration,
                            *default_control,
                        )

        self._impl_control_jacobian = [
            lambdify(
                self._state_model.arglist,
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
                default_calibration = np.zeros((self.calibration_size, 1))
                default_control = np.zeros((self.control_size, 1))
                for jit_impl in self._impl_control_jacobian:
                    for _i in range(5):
                        jit_impl(
                            default_dt,
                            *default_state,
                            *default_calibration,
                            *default_control,
                        )

    def _construct_sensors(
        self, state_model, sensor_models, sensor_noises, calibration_map, config
    ):
        assert set(sensor_models.keys()) == set(sensor_noises.keys())

        self.params["sensor_models"] = {
            k: SensorModel(
                state_model=state_model,
                sensor_model=model,
                calibration_map=calibration_map,
                config=config,
            )
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
                        for _i in range(5):
                            jit_impl(*default_state)

            self._impl_sensor_jacobians[k] = impl_sensor_jacobian

        self.innovations = {}
        self.sensor_prediction_uncertainty = {}

    def process_jacobian(self, dt, state, control):
        jacobian = np.zeros((self.state_size, self.state_size))
        for row in range(self.state_size):
            for col in range(self.state_size):
                jacobian[row, col] = self._impl_process_jacobian[
                    row * self.state_size + col
                ](dt, *state, *self.calibration_vector, *control)
        return jacobian

    def control_jacobian(self, dt, state, control):
        jacobian = np.zeros((self.state_size, self.control_size))
        for row in range(self.state_size):
            for col in range(self.control_size):
                jacobian[row, col] = self._impl_control_jacobian[
                    row * self.control_size + col
                ](dt, *state, *self.calibration_vector, *control)
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

    def process_model(self, dt, state, covariance, control=None):
        if control is None:
            control = np.zeros((0, 1))

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

        next_state = self._state_model.model(dt, state, control)
        assert next_state.shape == state.shape

        return StateAndCovariance(next_state, next_covariance)

    def sensor_model(self, sensor_key, state, covariance, sensor_reading):
        model_impl = self.params["sensor_models"][sensor_key]
        sensor_size = len(model_impl.readings)
        Q_t = _model_noise = self.params["sensor_noises"][sensor_key]

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

        K_t = _kalman_gain = np.matmul(covariance, np.matmul(H_t.transpose(), S_inv))

        self.innovations[sensor_key] = innovation = sensor_reading - expected_reading

        next_covariance = covariance - np.matmul(K_t, np.matmul(H_t, covariance))
        assert next_covariance.shape == covariance.shape

        next_state = state + np.matmul(K_t, innovation)
        assert next_state.shape == state.shape

        return StateAndCovariance(next_state, next_covariance)

    ### scikit-learn / sklearn interface ###

    def _flatten_scoring_params(self, params):
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
        assert self.params["process_noise"] is not None
        assert self.params["sensor_models"] is not None
        assert self.params["sensor_noises"] is not None

        x0 = self._flatten_scoring_params(self.params)

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
            raise MinimizationFailure(result)

        soln_as_params = self._inverse_flatten_scoring_params(result.x)
        self.set_params(**soln_as_params)

        return self

    # Compute the squared Mahalanobis distances of given observations.
    def mahalanobis(self, X):
        innovations, states, covariances = self.transform(X, include_states=True)
        n_samples, n_sensors = innovations.shape

        innovations = np.array(innovations).reshape((n_samples, n_sensors, 1))

        return innovations.flatten()

    # Compute something like the log-likelihood of X_test under the estimated Gaussian model.
    def score(self, X, y=None, sample_weight=None, explain_score=False):
        mahalanobis_distance_squared = self.mahalanobis(X)
        normalized_innovations = np.sqrt(mahalanobis_distance_squared)

        if sample_weight is None:
            avg = np.sum(np.square(np.average(normalized_innovations)))
            var = np.sum(mahalanobis_distance_squared)
        else:
            avg = np.sum(
                np.square(np.average(normalized_innovations, weights=sample_weight))
            )
            var = np.sum(mahalanobis_distance_squared * sample_weight)

        # bias->0
        bias_weight = 1e1
        bias_score = avg

        # variance->1
        # minima at var = 1, innovations match noise model
        variance_weight = 1e0
        variance_score = (1.0 / var + var) / 2.0

        # prefer smaller matrix terms
        matrix_weight = 1e-2
        matrix_score = np.sum(np.square(self.params["process_noise"]))
        for sensor_noise in self.params["sensor_noises"].values():
            matrix_score += np.sum(np.square(sensor_noise))

        if explain_score:
            return (
                (
                    bias_weight * bias_score
                    + variance_weight * variance_score
                    + matrix_weight * matrix_score
                ),
                (
                    bias_weight,
                    bias_score,
                    variance_weight,
                    variance_score,
                    matrix_weight,
                    matrix_score,
                ),
            )

        return (
            bias_weight * bias_score
            + variance_weight * variance_score
            + matrix_weight * matrix_score
        )

    # Transform readings to innovations
    def transform(self, X, include_states=False):
        n_samples, n_features = X.shape

        dt = 0.1

        state = np.zeros((self.state_size, 1))
        covariance = np.eye(self.state_size)

        innovations = []
        states = [state]
        covariances = [covariance]

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
                # Mahalanobis distance = sqrt((x - u).T * S^{-1} * (x - u))
                # for:
                #   u: predicted sensor readings
                #   x: sensor readings
                #   S: predicted sensor variance
                innovation.append(
                    float(
                        np.matmul(
                            np.matmul(
                                self.innovations[key].T,
                                np.linalg.inv(self.sensor_prediction_uncertainty[key]),
                            ),
                            self.innovations[key],
                        )
                    )
                )

            states.append(state)
            covariances.append(covariance)
            innovations.append(innovation)
            assert innovations[-1] is not None

        # minima at x = 1, innovations match noise model
        if include_states:
            return np.array(innovations), np.array(states), np.array(covariances)

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


@dataclass
class Config:
    compile: bool = False
    warm_jit: bool = compile
    common_subexpression_elimination: bool = True
    python_modules = DEFAULT_MODULES
    extra_validation: bool = False


def compile(symbolic_model, calibration_map=None, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    return Model(
        symbolic_model=symbolic_model, calibration_map=calibration_map, config=config
    )


def compile_ekf(
    state_model,
    process_noise,
    sensor_models,
    sensor_noises,
    calibration_map=None,
    *,
    config=None,
):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    common.model_validation(
        state_model,
        process_noise,
        sensor_models,
        extra_validation=config.extra_validation,
    )

    return ExtendedKalmanFilter(
        state_model=state_model,
        process_noise=process_noise,
        sensor_models=sensor_models,
        sensor_noises=sensor_noises,
        calibration_map=calibration_map,
        config=config,
    )
