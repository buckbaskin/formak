"""
Feature Test.

Generate a Python model and run it against a handful of tests that exercise a
scikit-learn-like interface

Passes if the interfaces return results that match the expected shape of each
scikit-learn interface
"""

import numpy as np

from formak.ui.model import Model as UiModel
from formak.runtime.sklearn import SklearnEKFAdapter
from sympy import Symbol

def test_UI_like_sklearn():
    dt = Symbol("dt")

    tp = _trajectory_properties = {k: Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = Symbol("thrust")

    state = set(tp.values())
    control = {thrust}

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    params = {
        "process_noise": {thrust: 1.0},
        "sensor_models": {"simple": {Symbol("v"): Symbol("v")}},
        "sensor_noises": {"simple": {Symbol("v"): 1.0}},
    }

    model = SklearnEKFAdapter(
        UiModel(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    # reading = [thrust, z, v]
    readings = _X = np.array(
        [
            [
                10,
                0,
            ],
            [
                10,
                0,
            ],
            [
                9,
                1,
            ],
        ]
    )
    n_samples, n_features = readings.shape
    n_control = len(control)
    n_readings = len(params["sensor_models"])
    assert n_features == n_control + n_readings

    # Fit the model to data
    assert isinstance(model.fit(readings), SklearnEKFAdapter)

    # Interface based on:
    #   - sklearn.covariance.EmpiricalCovariance https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance.fit

    # Compute the squared Mahalanobis distances of given observations.
    assert model.mahalanobis(readings).shape == (n_samples,)
    # Compute something like the log-likelihood of X_test under the estimated Gaussian model.
    assert isinstance(model.score(readings), float)

    # Interface based on:
    #   - sklearn.manifold.LocallyLinearEmbedding https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding

    # Transform readings to innovations
    assert model.transform(readings).shape == (n_samples, n_readings)
    # Fit the model to data and transform readings to innovations
    assert model.fit_transform(readings).shape == (n_samples, n_readings)

    # Get parameters for this estimator.
    assert isinstance(model.get_params(deep=True), dict)
    # Set the parameters of this estimator.
    assert isinstance(model.set_params(**params), SklearnEKFAdapter)
