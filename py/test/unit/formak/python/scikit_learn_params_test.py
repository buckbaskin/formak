import numpy as np

from formak import python, ui


def test_get_params():
    dt = ui.Symbol("dt")

    tp = _trajectory_properties = {k: ui.Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = ui.Symbol("thrust")

    state = set(tp.values())
    control = {thrust}

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
    readings = np.array([[10, 0, 0], [10, 0, 1], [9, 1, 2]])
    n_samples, n_features = readings.shape

    # Get parameters for this estimator.
    assert isinstance(model.get_params(deep=True), dict)
    # Set the parameters of this estimator.
    assert isinstance(model.set_params(**params), python.ExtendedKalmanFilter)


def test_set_params():
    dt = ui.Symbol("dt")

    tp = _trajectory_properties = {k: ui.Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = ui.Symbol("thrust")

    state = set(tp.values())
    control = {thrust}

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
    readings = np.array([[10, 0, 0], [10, 0, 1], [9, 1, 2]])
    n_samples, n_features = readings.shape

    # Set the parameters of this estimator.
    assert isinstance(model.set_params(**params), python.ExtendedKalmanFilter)
