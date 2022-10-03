import numpy as np
import pytest

from formak import ui, python


def test_UI_like_sklearn():
    dt = ui.Symbol("dt")

    tp = trajectory_properties = {k: ui.Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = ui.Symbol("thrust")

    state = set(tp.values())
    control = set([thrust])

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    params = {
        "process_noise": np.eye(1),
        "sensor_models": {"simple": {ui.Symbol("v"): ui.Symbol("v")}},
        "sensor_noises": {"simple": np.eye(1)},
    }

    model = python.compile_ekf(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    # reading = [thrust, z, v]
    readings = X = np.array([[10, 0, 0], [10, 0, 1], [9, 1, 2]])
    n_samples, n_features = readings.shape

    # Fit the model to data
    assert isinstance(model.fit(readings), python.ExtendedKalmanFilter)

    # Interface based on:
    #   - sklearn.covariance.EmpiricalCovariance https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance.fit

    # Compute the squared Mahalanobis distances of given observations.
    assert model.mahalanobis(readings).shape == (n_samples,)
    # Compute the log-likelihood of X_test under the estimated Gaussian model.
    assert isinstance(model.score(readings), float)

    # Interface based on:
    #   - sklearn.manifold.LocallyLinearEmbedding https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding

    # Transform readings to innovations
    assert model.transform(readings).shape == (n_samples, n_features - len(control))
    # Fit the model to data and transform readings to innovations
    assert model.fit_transform(readings).shape == (n_samples, n_features - len(control))

    # Get parameters for this estimator.
    assert isinstance(model.get_params(deep=True), dict)
    # Set the parameters of this estimator.
    assert isinstance(model.set_params(**params), python.ExtendedKalmanFilter)
