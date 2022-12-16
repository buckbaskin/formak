import numpy as np

from formak import ui, python


def test_fit_transform():
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
    n_samples, n_features = readings.shape

    result = model.fit_transform(readings)
    assert result.shape == (n_samples, n_features - len(control))

    assert not np.allclose(result, np.zeros_like(result))
    assert np.all(np.abs(result[2:]) < np.abs(readings[2:, 1:]))
