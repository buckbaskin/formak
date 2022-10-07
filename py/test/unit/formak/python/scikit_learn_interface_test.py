import numpy as np
import pytest

from numpy.random import default_rng
from formak import ui, python


def test_fit():
    random = default_rng(1)

    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = set([x])
    control = set([v])

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": np.eye(1),
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": np.eye(1)},
    }

    model = python.compile_ekf(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    true_variance = 2.0

    # reading = [v, x]
    readings = X = np.array(
        [[0, random.normal(scale=true_variance)] for _ in range(20)]
    )

    # Fit the model to data
    result = model.fit(readings)
    assert isinstance(result, python.ExtendedKalmanFilter)

    params = dict(model.get_params())
    assert params["process_noise"][0, 0] < 1.0
    assert np.isclose(params["sensor_noises"]["simple"][0, 0], true_variance)


def test_mahalanobis():
    random = default_rng(1)

    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = set([x])
    control = set([v])

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": np.eye(1),
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": np.eye(1)},
    }

    model = python.compile_ekf(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    true_variance = 2.0

    # reading = [v, x]
    readings = X = np.array([[0, 2], [0, -2], [0, 1], [0, -1], [0, 0.5], [0, -0.5]])

    result = model.mahalanobis(readings)
    assert result.shape == (6,)

    assert not np.allclose(result, np.zeros_like(result))
    assert np.allclose(result, [2, -2, 1, -1, 0.5, -0.5])


def test_score():
    random = default_rng(1)

    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = set([x])
    control = set([v])

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": np.eye(1),
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": np.eye(1)},
    }

    model = python.compile_ekf(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    true_variance = 2.0

    # reading = [v, x]
    readings = X = np.array([[0, 2], [0, -2], [0, 1], [0, -1], [0, 0.5], [0, -0.5]])

    score = model.score(readings)
    assert isinstance(score, float)

    assert not np.allclose(score, 0.0)

    # score lower for lower variance
    assert score > model.score(readings * 0.5)
    # score higher for higher variance
    assert score < model.score(readings * 3.0)

    # score lower for lower bias
    assert model.score(readings * 0.5) < model.score(readings * 0.5 + 2.0)


def test_transform():
    random = default_rng(1)

    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = set([x])
    control = set([v])

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": np.eye(1),
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": np.eye(1)},
    }

    model = python.compile_ekf(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    true_variance = 2.0

    # reading = [v, x]
    readings = X = np.array([[0, 2], [0, -2], [0, 1], [0, -1], [0, 0.5], [0, -0.5]])
    n_samples, n_features = readings.shape

    result = model.transform(readings)
    assert result.shape == (n_samples, n_features - len(control))

    assert not np.allclose(result, np.zeros_like(result))
    assert np.allclose(result, readings[:, 1])


def test_fit_transform():
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

    result = model.fit_transform(readings)
    assert result.shape == (n_samples, n_features - len(control))

    assert not np.allclose(result, np.zeros_like(result))
    assert np.allclose(result, readings[:, 1])


def test_get_params():
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

    # Get parameters for this estimator.
    assert isinstance(model.get_params(deep=True), dict)
    # Set the parameters of this estimator.
    assert isinstance(model.set_params(**params), python.ExtendedKalmanFilter)


def test_set_params():
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

    # Set the parameters of this estimator.
    assert isinstance(model.set_params(**params), python.ExtendedKalmanFilter)
