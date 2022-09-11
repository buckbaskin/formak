import numpy as np

from numpy.testing import assert_almost_equal
from formak import ui, python

def test_EKF_impl_control():
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
