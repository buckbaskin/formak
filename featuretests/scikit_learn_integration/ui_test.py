import numpy as np
import pytest

from formak.ui import *


def test_UI_like_sklearn():
    dt = Symbol("dt")

    tp = trajectory_properties = {k: Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = Symbol("thrust")

    state = set(tp.values())
    control = set([thrust])

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    model = Model(dt=dt, state=state, control=control, state_model=state_model)

    # reading = [thrust, z, v]
    readings = X = np.array([[10, 0, 0], [10, 0, 1], [9, 1, 2]])
    n_samples, n_features = readings.shape

    # Fit the model to data
    model.fit(readings)

    # Interface based on:
    #   - sklearn.covariance.EmpiricalCovariance https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance.fit

    # Compute the squared Mahalanobis distances of given observations.
    assert model.mahalanobis(readings).shape == (n_samples,)
    # Compute the log-likelihood of X_test under the estimated Gaussian model.
    assert isinstance(model.score(readings), float)

    # Interface based on:
    #   - sklearn.manifold.LocallyLinearEmbedding https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding

    # Transform readings to innovations
    model.transform(readings)
    # Fit the model to data and transform readings to innovations
    model.fit_transform(readings)

    # Get parameters for this estimator.
    model.get_params(deep=True)
    # Set the parameters of this estimator.
    set_params(**params)
