import numpy as np

from formak import python, ui


def test_score():
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

    # reading = [v, x]
    readings = np.array([[0, 2], [0, -2], [0, 1], [0, -1], [0, 0.5], [0, -0.5]])

    score = model.score(readings)
    assert isinstance(score, float)

    assert not np.allclose(score, 0.0)

    # score lower for lower variance
    assert score > model.score(readings * 0.5)
    # score higher for higher variance
    assert score < model.score(readings * 3.0)

    # score lower for lower bias
    assert model.score(readings * 0.5) < model.score(readings * 0.5 + 2.0)


def test_score_two_sensor():
    dt = ui.Symbol("dt")

    x, v, a = ui.symbols(["x", "v", "a"])

    state = {x, v}
    control = {a}

    state_model = {
        x: x + dt * v,
        v: v + dt * a,
    }

    params = {
        "process_noise": {ui.Symbol("a"): 1.0},
        "sensor_models": {"position": {x: x}, "velocity": {v: v}},
        "sensor_noises": {"position": {x: 1}, "velocity": {v: 1}},
    }

    model = python.SklearnEKFAdapter(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    # reading = [v, x]
    readings = np.array(
        [[0, 2, 0], [0, -2, 0], [0, 1, 0], [0, -1, 0], [0, 0.5, 0], [0, -0.5, 0]]
    )

    score = model.score(readings)
    assert isinstance(score, float)

    assert not np.allclose(score, 0.0)

    # score lower for lower variance
    assert score > model.score(readings * 0.5)
    # score higher for higher variance
    assert score < model.score(readings * 3.0)

    # score lower for lower bias
    assert model.score(readings * 0.5) < model.score(readings * 0.5 + 2.0)


def test_score_two_sensor_explained():
    dt = ui.Symbol("dt")

    x, v, a = ui.symbols(["x", "v", "a"])

    state = {x, v}
    control = {a}

    state_model = {
        x: x + dt * v,
        v: v + dt * a,
    }

    params = {
        "process_noise": {ui.Symbol("a"): 1.0},
        "sensor_models": {"position": {x: x}, "velocity": {v: v}},
        "sensor_noises": {"position": {x: 1}, "velocity": {v: 1}},
    }

    model = python.SklearnEKFAdapter(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    # reading = [v, x]
    readings = np.array(
        [[0, 2, 0], [0, -2, 0], [0, 1, 0], [0, -1, 0], [0, 0.5, 0], [0, -0.5, 0]]
    )

    score, score_components = model.score(readings, explain_score=True)
    assert isinstance(score, float)

    assert not np.allclose(score, 0.0)

    # score lower for lower variance
    assert score > model.score(readings * 0.5)
    # score higher for higher variance
    assert score < model.score(readings * 3.0)

    # score lower for lower bias
    assert model.score(readings * 0.5) < model.score(readings * 0.5 + 2.0)
