import numpy as np
import pytest
from numpy.random import default_rng
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

from formak import python, ui


def test_fit():
    random = default_rng(1)

    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = {x}
    control = {v}

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": {v: 1.0},
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": {x: 1}},
    }

    model = python.SklearnEKFAdapter(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    true_variance = 2.0

    # reading = [v, x]
    readings = np.array([[0, random.normal(scale=true_variance)] for _ in range(20)])

    pre_score = model.score(readings)

    # Fit the model to data
    result = model.fit(readings)
    assert isinstance(result, python.SklearnEKFAdapter)

    post_score = model.score(readings)

    assert pre_score > post_score


def test_clone():
    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = {x}
    control = {v}

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": {v: 1.0},
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": {x: 1}},
    }

    model = python.SklearnEKFAdapter(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    clone(model)


@pytest.mark.xfail(reason="WIP on implementing various checks")
def test_estimator_against_sklearn_checks():
    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = {x}
    control = {v}

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": {v: 1.0},
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": {x: 1}},
    }

    model = python.SklearnEKFAdapter(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    check_estimator(model)
