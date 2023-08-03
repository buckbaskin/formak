import warnings

import numpy as np
import pytest
from formak.exceptions import ModelConstructionError
from numpy.testing import assert_almost_equal

from formak import python, ui

warnings.filterwarnings("error")


def test_EKF_model_collapse():
    config = python.Config()
    config.extra_validation = True

    with pytest.raises(ModelConstructionError):
        python.compile_ekf(
            state_model=ui.Model(
                ui.Symbol("dt"),
                set(ui.symbols(["x", "y"])),
                set(ui.symbols(["a"])),
                {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
            ),
            process_noise={ui.Symbol("a"): 1.0},
            sensor_models={},
            sensor_noises={},
            config=config,
        )


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
        process_noise={ui.Symbol("a"): 1.0},
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
        process_noise={ui.Symbol("a"): 1.0},
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
        state=state_vector,
        covariance=covariance,
        sensor_key="simple",
        sensor_reading=np.array([[reading]]),
    )
    assert abs(reading - next_state[0]) < abs(reading - state_vector[0])

    next_state, next_cov = ekf.sensor_model(
        state=state_vector,
        covariance=covariance,
        sensor_key="combined",
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
        process_noise={ui.Symbol("a"): 1.0},
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


def test_SensorModel_calibration():
    config = python.Config()

    def read_once(calibration_map):
        model = python.SensorModel(
            state_model=ui.Model(
                ui.Symbol("dt"),
                set(ui.symbols(["x"])),
                set(),
                {ui.Symbol("x"): "x"},
                calibration=set(ui.symbols(["a"])),
            ),
            calibration_map=calibration_map,
            sensor_model={"reading": ui.Symbol("a") + ui.Symbol("x")},
            config=config,
        )

        return model.model(np.zeros((1, 1)))

    calibration = {
        "a": -1.4,
    }
    calibration_map = {ui.Symbol(k): v for k, v in calibration.items()}

    assert read_once(calibration_map) == -1.4

    calibration = {
        "a": 1.2,
    }
    calibration_map = {ui.Symbol(k): v for k, v in calibration.items()}

    assert read_once(calibration_map) == 1.2


def test_EKF_sensor_jacobian_calibration():
    config = python.Config()

    def read_once(calibration_map):
        ekf = python.ExtendedKalmanFilter(
            state_model=ui.Model(
                ui.Symbol("dt"),
                set(ui.symbols(["x"])),
                set(),
                {ui.Symbol("x"): "x"},
                calibration=set(ui.symbols(["a"])),
            ),
            process_noise={},
            calibration_map=calibration_map,
            sensor_models={"key": {"reading": ui.Symbol("a") * ui.Symbol("x")}},
            sensor_noises={"key": np.eye(1)},
            config=config,
        )

        return ekf.sensor_jacobian("key", np.zeros((1, 1)))[0, 0]

    calibration = {
        "a": -1.4,
    }
    calibration_map = {ui.Symbol(k): v for k, v in calibration.items()}

    assert read_once(calibration_map) == -1.4

    calibration = {
        "a": 1.2,
    }
    calibration_map = {ui.Symbol(k): v for k, v in calibration.items()}

    assert read_once(calibration_map) == 1.2
