"""
Feature Test.

Fit a Python model via the scikit-learn interface.

Passes if the fit model scores better than the unfit model
"""

import numpy as np

from formak import python, ui


def test_like_sklearn_regression():
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
        "process_noise": {ui.Symbol("thrust"): 1.0},
        "sensor_models": {
            "z": {ui.Symbol("z"): ui.Symbol("z")},
            "v": {ui.Symbol("v"): ui.Symbol("v")},
        },
        "sensor_noises": {"z": {ui.Symbol("z"): 1.0}, "v": {ui.Symbol("v"): 1.0}},
    }
    model = python.SklearnEKFAdapter(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    # reading = [thrust, z, v]
    readings = _X = np.array([[10, 0, 0], [10, 0, 1], [9, 1, 2], [8, 1, 2], [7, 1, 2]])

    unfit_score = model.score(readings)

    assert unfit_score > 1.0  # Don't accidentally start with a perfect model

    # Fit the model to data
    model.fit(readings)

    fit_score = model.score(readings)

    assert fit_score < unfit_score
