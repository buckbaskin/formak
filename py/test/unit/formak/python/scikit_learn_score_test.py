import numpy as np

from formak import ui, python


def test_score():
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
