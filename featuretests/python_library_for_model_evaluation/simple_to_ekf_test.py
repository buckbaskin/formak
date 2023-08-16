import numpy as np

from formak import python, ui


def test_ekf_simple():
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

    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

    v = ui.Symbol("v")

    python_ekf = python.compile_ekf(
        state_model=model,
        process_noise={thrust: 1.0},
        sensor_models={"simple": {v: v}},
        sensor_noises={"simple": {v: 1.0}},
    )
    assert isinstance(python_ekf, python.ExtendedKalmanFilter)

    state_vector = np.array([[0.0, 0.0, 0.0, 0.0]]).transpose()
    state_variance = np.eye(4)
    control_vector = np.array([[0.0]])

    state_vector_next, state_variance_next = python_ekf.process_model(
        0.1, state_vector, state_variance, control_vector
    )

    state_variance_next, state_variance_next = python_ekf.sensor_model(
        state=state_vector,
        covariance=state_variance,
        sensor_key="simple",
        sensor_reading=np.array([[0.0]]),
    )
