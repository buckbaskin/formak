import numpy as np

from formak import python, ui


def make_ekf(calibration_map):
    dt = ui.Symbol("dt")

    state = ui.Symbol("state")

    control_velocity = ui.Symbol("control_velocity")
    calibration_velocity = ui.Symbol("calibration_velocity")

    state_model = {state: state + dt * (control_velocity + calibration_velocity)}

    state_set = {state}
    control_set = {control_velocity}

    model = ui.Model(
        dt=dt,
        state=state_set,
        control=control_set,
        state_model=state_model,
        calibration={calibration_velocity},
    )

    ekf = python.compile_ekf(
        state_model=model,
        process_noise={control_velocity: 1.0},
        sensor_models={"simple": {state: state}},
        sensor_noises={"simple": {state: 1e-9}},
        calibration_map=calibration_map,
        config={"innovation_filtering": 3.0},
    )
    return ekf


def test_constructor():
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map)

    innovation = np.zeros((1, 1))
    S_inv = np.eye(1)

    assert not ekf.remove_innovation(innovation, S_inv)

    innovation = np.array([[4.0]])
    S_inv = np.eye(1)

    assert ekf.remove_innovation(innovation, S_inv)

    innovation = np.array([[-4.0]])
    S_inv = np.eye(1)

    assert ekf.remove_innovation(innovation, S_inv)
