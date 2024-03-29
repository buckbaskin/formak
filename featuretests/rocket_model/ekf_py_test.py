"""
Feature Test.

Define a Python implementation of a rocket model. This previously demonstrated
some performance limitations for the Python generation with a bigger model.

Passes if the Python implementation is created and runs without an exception
"""

from itertools import repeat

from model_definition import (
    model_definition,
    named_acceleration,
    named_rotation_rate,
    named_translation,
)

from formak import python, ui


def test_python_EKF():
    definition = model_definition()
    model = definition["model"]

    calibration = {
        # orientation would need to invert a rotation matrix
        "IMU_ori_pitch": 0.0,
        "IMU_ori_roll": 0.0,
        "IMU_ori_yaw": 0.0,
        # pos_IMU_from_CON_in_CON     [m]     [-0.08035, 0.28390, -1.42333 ]
        "IMU_pos_x": -0.08035,
        "IMU_pos_y": 0.28390,
        "IMU_pos_z": -1.42333,
    }
    calibration_map = {ui.Symbol(k): v for k, v in calibration.items()}

    (reading_orientation_rate_states, _) = named_rotation_rate("IMU_reading")
    reading_acceleration_states = sorted(
        named_acceleration("IMU_reading").free_symbols, key=lambda x: x.name
    )

    process_noise = {
        k: v
        for k, v in list(zip(reading_orientation_rate_states, repeat(0.1)))
        + list(zip(reading_acceleration_states, repeat(1.0)))
    }

    CON_position_in_global_frame = named_translation("CON_pos")

    python_implementation = python.compile_ekf(
        symbolic_model=model,
        process_noise=process_noise,
        sensor_models={
            "altitude": {ui.Symbol("altitude"): CON_position_in_global_frame[2]}
        },
        sensor_noises={"altitude": {ui.Symbol("altitude"): 1.0}},
        calibration_map=calibration_map,
    )

    state_vector = python_implementation.State()
    state_covariance = python_implementation.Covariance()
    control_vector = python_implementation.Control()

    _state_vector_next = python_implementation.process_model(
        dt=0.01, state=state_vector, covariance=state_covariance, control=control_vector
    )
