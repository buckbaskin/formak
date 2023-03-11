import warnings

import numpy as np
from numpy.testing import assert_almost_equal

from formak import python, ui

warnings.filterwarnings("error")


def test_EKF_process_with_control():
    config = python.Config()
    dt = 0.1

    ekf = python.ExtendedKalmanFilter(
        state_model=ui.Model(
            ui.Symbol("dt"),
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
        ),
        process_noise=np.eye(1),
        sensor_models={},
        sensor_noises={},
        config=config,
    )

    control_vector = np.array([[0.2]])
    covariance = np.eye(2)

    state_vector = np.array([[0.0, 0.0]]).transpose()
    assert_almost_equal(
        ekf.process_model(
            dt=dt, state=state_vector, covariance=covariance, control=control_vector
        )[0].transpose(),
        [[0.0, 0.02]],
    )

    state_vector = np.array([[0.0, 1.0]]).transpose()
    assert_almost_equal(
        ekf.process_model(
            dt=dt, state=state_vector, covariance=covariance, control=control_vector
        )[0].transpose(),
        [[0.0, 1.02]],
    )

    state_vector = np.array([[1.0, 0.0]]).transpose()
    assert_almost_equal(
        ekf.process_model(
            dt=dt, state=state_vector, covariance=covariance, control=control_vector
        )[0].transpose(),
        [[0.0, 0.02]],
    )

    state_vector = np.array([[1.0, 1.0]]).transpose()
    assert_almost_equal(
        ekf.process_model(
            dt=dt, state=state_vector, covariance=covariance, control=control_vector
        )[0].transpose(),
        [[1.0, 1.02]],
    )


def test_EKF_sensor():
    config = python.Config()

    ekf = python.ExtendedKalmanFilter(
        state_model=ui.Model(
            ui.Symbol("dt"),
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
        ),
        process_noise=np.eye(1),
        sensor_models={
            "simple": {"reading1": ui.Symbol("x")},
            "combined": {"reading2": ui.Symbol("x") + ui.Symbol("y")},
        },
        sensor_noises={"simple": np.eye(1), "combined": np.eye(1)},
        config=config,
    )

    covariance = np.eye(2)
    reading = 1.0
    state_vector = np.array([[0.0, 0.0]]).transpose()

    next_state, next_cov = ekf.sensor_model(
        "simple",
        state=state_vector,
        covariance=covariance,
        sensor_reading=np.array([[reading]]),
    )
    assert abs(reading - next_state[0]) < abs(reading - state_vector[0])

    next_state, next_cov = ekf.sensor_model(
        "combined",
        state=state_vector,
        covariance=covariance,
        sensor_reading=np.array([[reading]]),
    )
    assert abs(reading - next_state[0]) < abs(reading - state_vector[0])
    assert abs(reading - next_state[1]) < abs(reading - state_vector[1])


def test_EKF_process_jacobian():
    config = python.Config()
    dt = 0.1

    ekf = python.ExtendedKalmanFilter(
        state_model=ui.Model(
            ui.Symbol("dt"),
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
        ),
        process_noise=np.eye(1),
        sensor_models={},
        sensor_noises={},
        config=config,
    )

    control_vector = np.array([[0.2]])

    state_vector = np.array([[0.0, 0.0]]).transpose()
    assert_almost_equal(
        ekf.process_jacobian(dt=dt, state=state_vector, control=control_vector),
        [[0.0, 0.0], [0.0, 1.0]],
    )

    state_vector = np.array([[0.0, 1.0]]).transpose()
    assert_almost_equal(
        ekf.process_jacobian(dt=dt, state=state_vector, control=control_vector),
        [[1.0, 0.0], [0.0, 1.0]],
    )

    state_vector = np.array([[1.0, 0.0]]).transpose()
    assert_almost_equal(
        ekf.process_jacobian(dt=dt, state=state_vector, control=control_vector),
        [[0.0, 1.0], [0.0, 1.0]],
    )

    state_vector = np.array([[1.0, 1.0]]).transpose()
    assert_almost_equal(
        ekf.process_jacobian(dt=dt, state=state_vector, control=control_vector),
        [[1.0, 1.0], [0.0, 1.0]],
    )
