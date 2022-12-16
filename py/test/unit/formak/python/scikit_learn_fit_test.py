import numpy as np

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
    readings = np.array([[0, random.normal(scale=true_variance)] for _ in range(20)])

    pre_score = model.score(readings)

    # Fit the model to data
    result = model.fit(readings)
    assert isinstance(result, python.ExtendedKalmanFilter)

    post_score = model.score(readings)

    assert pre_score > post_score

