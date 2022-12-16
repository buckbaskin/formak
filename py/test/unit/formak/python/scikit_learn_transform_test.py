import numpy as np

from formak import ui, python


def test_transform():
    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = set([x])
    control = set([v])

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": np.eye(1) * 1e-6,
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": np.eye(1)},
    }

    model = python.compile_ekf(
        ui.Model(dt=dt, state=state, control=control, state_model=state_model), **params
    )

    # reading = [v, x]
    readings = np.array([[0, 2], [0, -2], [0, 1], [0, -1], [0, 0.5], [0, -0.5]])
    n_samples, n_features = readings.shape

    result = model.transform(readings)
    assert result.shape == (n_samples, n_features - len(control))

    assert not np.allclose(result, np.zeros_like(result))
    assert np.allclose(result[2:], np.square(readings[2:, 1:]), atol=0.3)


def test_transform_kalman_filter_args():
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
    readings = X = np.array([[0, 2], [0, -2], [0, 1], [0, -1], [0, 0.5], [0, -0.5]])
    n_samples, n_features = readings.shape
    innovations, states, covariances = model.transform(X, include_states=True)

    assert innovations.shape == (n_samples, n_features - len(control))
    assert states.shape == (n_samples + 1, len(state), 1)
    assert covariances.shape == (n_samples + 1, len(state), len(state))

    assert not np.allclose(innovations, np.zeros_like(innovations))
    assert not np.allclose(states, np.zeros_like(states))
    assert not np.allclose(covariances, np.zeros_like(covariances))
