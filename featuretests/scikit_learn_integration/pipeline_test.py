"""
Feature Test.

Passes if running a model in a pipeline doesn't raise exceptions
"""
import numpy as np
from sklearn.pipeline import Pipeline

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
        "sensor_models": {"simple": {ui.Symbol("v"): ui.Symbol("v")}},
        "sensor_noises": {"simple": {ui.Symbol("v"): 1.0}},
    }
    model = python.SklearnAdapter(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    estimators = [("formak model", model)]
    pipeline = Pipeline(estimators)

    # reading = [thrust, z, v]
    readings = _X = np.array([[10, 0, 0], [10, 0, 1], [9, 1, 2]])
    n_samples, n_features = readings.shape

    # Fit the pipeline to data
    pipeline.fit(readings)

    # Score the data
    pipeline.score(readings)
