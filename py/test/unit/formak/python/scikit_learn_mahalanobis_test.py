import numpy as np

from formak import python, ui


def test_mahalanobis():
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

    innovations, states, covariances = model.transform(readings, include_states=True)

    result = model.mahalanobis(readings)
    assert result.shape == (6,)

    assert not np.allclose(result, np.zeros_like(result))

    for i in range(len(innovations)):
        expected_distance = innovations[i]
        assert np.allclose(expected_distance, result[i])
