import numpy as np

from formak import python, ui


def test_transform():
    dt = ui.Symbol("dt")

    x, v = ui.symbols(["x", "v"])

    state = {x}
    control = {v}

    state_model = {
        x: x + dt * v,
    }

    params = {
        "process_noise": {v: 1e-6},
        "sensor_models": {"simple": {x: x}},
        "sensor_noises": {"simple": {x: 1}},
    }

    model = python.SklearnEKFAdapter(
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
    readings = X = np.array([[0, 2], [0, -2], [0, 1], [0, -1], [0, 0.5], [0, -0.5]])
    n_samples, n_features = readings.shape
    innovations, states, covariances = model.transform(X, include_states=True)

    assert innovations.shape == (n_samples, n_features - len(control))
    assert states.shape == (n_samples + 1,)
    assert isinstance(states[0], model.State)
    assert covariances.shape == (n_samples + 1,)
    assert isinstance(covariances[0], model.Covariance)

    assert not np.allclose(innovations, np.zeros_like(innovations))
    print('states', len(states))
    for s in states:
        print('data:', s.data)
    assert not np.allclose(
        [s.data for s in states], np.zeros((n_samples + 1, len(state), 1))
    )
    assert not np.allclose(
        [c.data for c in covariances], np.zeros((n_samples + 1, len(state), len(state)))
    )
