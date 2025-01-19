from __future__ import annotations

import dataclasses
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

import sympy
from backend_py.config import Config
from backend_py.extended_kalman_filter import (
    ExtendedKalmanFilter,
    assert_valid_covariance,
    nearest_positive_definite,
)
from backend_py.model import Model
from backend_py.named_vector import named_vector
from backend_py.sensor_model import SensorModel
from formak.exceptions import MinimizationFailure, ModelConstructionError
from frontend.model_validation import model_validation
from problemdefinition.ui_model_base import UiModelBase
from sympy import Symbol


def compile(symbolic_model, calibration_map=None, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    model_validation(
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
    symbolic_model: UiModelBase,
    process_noise: dict[Symbol | tuple[Symbol, Symbol], float],
    sensor_models: dict[str, sympy.core.expr.Expr],
    sensor_noises,
    calibration_map: dict[Symbol, float] | None = None,
    *,
    config=None,
) -> ExtendedKalmanFilter:
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    model_validation(
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
        symbolic_model: UiModelBase,
        process_noise: dict[Symbol | tuple[Symbol, Symbol], float],
        sensor_models: dict[str, sympy.core.expr.Expr],
        sensor_noises: dict[str, dict[Symbol | tuple[Symbol, Symbol], float]],
        calibration_map: dict[Symbol, float] | None = None,
        *,
        config: Config | None = None,
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
        symbolic_model: UiModelBase | None = None,
        process_noise: dict[Symbol | tuple[Symbol, Symbol], float] | None = None,
        sensor_models: dict[Symbol, sympy.core.expr.Expr] | None = None,
        sensor_noises: dict[Symbol | tuple[Symbol, Symbol], float] | None = None,
        calibration_map: dict[Symbol, float] | None = None,
        *,
        config: Config | None = None,
    ):
        self.symbolic_model = symbolic_model
        self.process_noise = process_noise
        self.sensor_models = sensor_models
        self.sensor_noises = sensor_noises
        self.calibration_map = calibration_map
        self.config = config

    def _flatten_process_noise(
        self, process_noise: dict[Symbol | tuple[Symbol, Symbol], float]
    ):
        for iSymbol in self.arglist_control:
            for jSymbol in self.arglist_control:
                if (iSymbol, jSymbol) in process_noise:
                    value = process_noise[(iSymbol, jSymbol)]
                elif (jSymbol, iSymbol) in process_noise:
                    value = process_noise[(jSymbol, iSymbol)]
                elif iSymbol == jSymbol and iSymbol in process_noise:
                    value = process_noise[iSymbol]
                else:
                    value = 0.0
                yield (iSymbol, jSymbol, value)

    def _sensor_noise_to_array(
        self,
        sensor_noises: dict[str, dict[Symbol | tuple[Symbol, Symbol], float]],
    ):
        matrix_sensor_noises = {}
        for key, model in self.sensor_models.items():
            assert isinstance(sensor_noises[key], dict)
            assert len(sensor_noises[key]) == len(self.sensor_models[key].keys())
            readings = sorted(list(model.keys()))
            ReadingCovariance = named_vector("ReadingCovariance", readings)

            assert_valid_covariance(sensor_noises[key], f"Sensor Noise [{key}]")
            matrix_sensor_noises[key] = ReadingCovariance.from_dict(sensor_noises[key])

        return matrix_sensor_noises

    def _compile_sensor_models(self, sensor_models: dict[str, sympy.core.expr.Expr]):
        return {
            k: SensorModel(
                state_model=self.state_model,
                sensor_model=model,
                calibration_map=self.calibration_map,
                config=self.config,
            )
            for k, model in sensor_models.items()
        }

    def _flatten_dict_diagonal(
        self,
        mapping: dict[Symbol | tuple[Symbol, Symbol], float],
        arglist: list[Symbol],
    ) -> Iterator[float]:
        for iSymbol in arglist:
            if (iSymbol, iSymbol) in mapping:
                value = mapping[(iSymbol, iSymbol)]
            elif iSymbol in mapping:
                value = mapping[iSymbol]
            else:
                value = 0.0
            yield value

    def _inverse_flatten_dict_diagonal(
        self, vector, arglist
    ) -> dict[Symbol | tuple[Symbol, Symbol], float]:
        for iIdx, iSymbol in enumerate(arglist):
            yield (iSymbol, vector[iIdx])

    def _flatten_scoring_params(self) -> list[float]:
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

        for _key, mapping in sorted(list(self.sensor_noises.items())):
            arglist = sorted(list(mapping.keys()))

            flattened.extend(self._flatten_dict_diagonal(mapping, arglist))

        return flattened

    def _inverse_flatten_scoring_params(self, flattened: list[float]) -> dict[str, Any]:
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

        params["process_noise"] = nearest_positive_definite(
            dict(self._inverse_flatten_dict_diagonal(controls, arglist_control))
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
    def fit(
        self, X: Any, y: Any | None = None, sample_weight: NDArray | None = None
    ) -> SklearnEKFAdapter:
        assert self.process_noise is not None
        assert self.sensor_models is not None
        assert self.sensor_noises is not None

        x0 = self._flatten_scoring_params()

        def minimize_this(x: NDArray) -> float:
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
    def mahalanobis(self, X: Any) -> NDArray:
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
            print("X\n", X.flatten())
            print("Innovations\n", innovations.flatten())
            raise AssertionError("innovations squared includes negative values")

        return innovations.flatten()

    # Compute something like the log-likelihood of X_test under the estimated Gaussian model.
    def score(
        self,
        X: Any,
        y: Any | None = None,
        sample_weight: Any | None = None,
        explain_score: bool = False,
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
            symbolic_model=self.symbolic_model,
            process_noise=self.process_noise,
            sensor_models=self.sensor_models,
            sensor_noises=self.sensor_noises,
            calibration_map=self.calibration_map,
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

        assert_valid_covariance(covariance.data)

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
            assert_valid_covariance(covariance.data)

            innovation = []

            for idx, key in enumerate(sorted(list(self.model_.sensor_models))):
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

                assert_valid_covariance(covariance.data)

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
                if np.any(self.model_.sensor_prediction_uncertainty[key] < 0.0):
                    print(idx, "key", key)
                    print(idx, "innovations")
                    print(self.model_.innovations[key])
                    print(idx, "uncertainty")
                    print(self.model_.sensor_prediction_uncertainty[key])
                    print(idx, "result")
                    print(innovation[-1])

            states.append(state)
            covariances.append(covariance)
            innovations.append(innovation)
            assert innovations[-1] is not None
            if np.any(np.array(innovations[-1]) < 0.0):
                raise AssertionError(
                    f"Negative assertion detected at index {len(innovations) - 1}. Value {innovations[-1]}"
                )

        innovations = np.array(innovations, dtype="float")
        if np.any(innovations < 0.0):
            print("Negative Innovation Detected")
            print(innovations[innovations < 0.0])
            raise AssertionError("All innovations should be non-negative.")

        # minima at x = 1, innovations match noise model
        if include_states:
            return (
                innovations,
                np.array(states),
                np.array(covariances),
            )

        return innovations

    # Fit the model to data and transform readings to innovations
    def fit_transform(self, X, y=None) -> NDArray | tuple[NDArray, NDArray, NDArray]:
        # TODO(buck): Implement the combined version (return innovations calculated while fitting)
        self.fit(X, y)
        return self.transform(X)

    # Get parameters for this estimator.
    def get_params(self, deep: bool = True) -> dict[str, Any]:
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

    def export_python(self) -> ExtendedKalmanFilter:
        return compile_ekf(
            self.symbolic_model,
            self.process_noise,
            self.sensor_models,
            self.sensor_noises,
            self.calibration_map,
            config=self.config,
        )
