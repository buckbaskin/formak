import numpy as np
import pytest

from formak import ui, python


def test_like_sklearn_regression():
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

    unfit_score = model.score(readings)

    assert unfit_score > 0.0 # Don't accidentally start with a perfect model

    # Fit the model to data
    model.fit(readings)

    fit_score = model.score(readings)

    assert fit_score < unfit_score
