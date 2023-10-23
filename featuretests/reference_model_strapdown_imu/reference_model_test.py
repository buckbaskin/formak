"""
Reference Model IMU.

This demonstrates an example of "playing" data from a NASA launch through the
IMU model to generate a motion trajectory in the global frame.

Data Sample Source
https://data.nasa.gov/Aerospace/Deorbit-Descent-and-Landing-Flight-1-DDL-F1-/vicw-ivgd
"""
import csv
import os
from math import hypot

import numpy as np
from formak.reference_models import strapdown_imu
from sympy import Matrix, Quaternion

from formak import python


def _line_to_control(line):
    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel

    rate = 50.0  # Hz
    dt = 1.0 / rate

    # TIME_NANOSECONDS_TAI,DATA_DELTA_VEL[1],DATA_DELTA_VEL[2],DATA_DELTA_VEL[3],DATA_DELTA_ANGLE[1],DATA_DELTA_ANGLE[2],DATA_DELTA_ANGLE[3]
    (
        time_ns,
        delta_vel_1,
        delta_vel_2,
        delta_vel_3,
        delta_angle_1,
        delta_angle_2,
        delta_angle_3,
    ) = line

    time_ns = int(float(time_ns))

    accel_1_ms2 = float(delta_vel_1) * rate
    accel_2_ms2 = float(delta_vel_2) * rate
    accel_3_ms2 = float(delta_vel_3) * rate

    body_rate_1 = float(delta_angle_1) * rate
    body_rate_2 = float(delta_angle_2) * rate
    body_rate_3 = float(delta_angle_3) * rate

    print("accel sum", hypot(accel_1_ms2, accel_2_ms2, accel_3_ms2))

    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel
    return time_ns, {
        imu_accel[0]: accel_1_ms2,
        imu_accel[1]: accel_2_ms2,
        imu_accel[2]: accel_3_ms2,
        imu_gyro[0]: body_rate_1,
        imu_gyro[1]: body_rate_2,
        imu_gyro[2]: body_rate_3,
    }


def stream_sample():
    filename = "NASA_sample.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        column_names = None
        for row in reader:
            if column_names is None:
                column_names = row
                continue

            time_ns, parsed_row = _line_to_control(row)

            yield time_ns, parsed_row


def stream_preignition():
    filename = "NASA_preignition.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        column_names = None
        for row in reader:
            if column_names is None:
                column_names = row
                continue

            time_ns, parsed_row = _line_to_control(row)

            yield time_ns, parsed_row


def starting_rotation():

    att_dcm_CON_IMU = np.array(
        [
            [-0.2477, -0.1673, 0.9543],
            [-0.0478, 0.9859, 0.1604],
            [-0.9677, -0.0059, -0.2522],
        ]
    )
    rotation = Quaternion.from_rotation_matrix(Matrix(att_dcm_CON_IMU))
    recreation = rotation.to_rotation_matrix(homogeneous=True)

    print("att")
    print(att_dcm_CON_IMU)
    print("rec")
    print(recreation)
    print("diff")
    print(att_dcm_CON_IMU - recreation)
    assert np.allclose(
        att_dcm_CON_IMU, np.array(recreation, dtype=np.float64), atol=1e-4
    )

    return rotation


def test_example_usage_of_reference_model():
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={strapdown_imu.g: 9.81},
    )
    assert imu is not None

    rate = 50  # Hz
    dt = 1.0 / rate

    orientation = starting_rotation()
    state = imu.State.from_dict(
        {
            "oriw": orientation.a,
            "orix": orientation.b,
            "oriy": orientation.c,
            "oriz": orientation.d,
        }
    )
    last_time = None

    TOL = 0.12

    # Stationary Data from a rocket launch pre-ignition
    for idx, (time_ns, control) in enumerate(stream_sample()):
        if last_time is not None:
            dt = (time_ns - last_time) * 1e-9
            print("timekeeper", idx, dt)
        last_time = time_ns

        control = imu.Control.from_dict(control)
        state = imu.model(dt, state, control)
        assert state is not None

        expected_state = imu.State.from_dict(
            {
                r"\ddot{x}_{A}_{1}": 0.0,
                r"\ddot{x}_{A}_{2}": 0.0,
                r"\ddot{x}_{A}_{3}": 0.0,
                r"\dot{\phi}": 0.0,
                r"\dot{\psi}": 0.0,
                r"\dot{\theta}": 0.0,
                r"\dot{x}_{A}_{1}": 0.0,
                r"\dot{x}_{A}_{2}": 0.0,
                r"\dot{x}_{A}_{3}": 0.0,
                "oriw": orientation.a,
                "orix": orientation.b,
                "oriy": orientation.c,
                "oriz": orientation.d,
                r"x_{A}_{1}": 0.0,
                r"x_{A}_{2}": 0.0,
                r"x_{A}_{3}": 0.0,
            }
        )

        if not np.allclose(state.data, expected_state.data, atol=TOL):
            state.render_diff(expected_state)
            1 / 0
        assert np.allclose(state.data, expected_state.data, atol=TOL)


def test_example_usage_of_reference_model_preignition():
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={strapdown_imu.g: 9.81},
    )
    assert imu is not None

    rate = 50  # Hz
    dt = 1.0 / rate

    orientation = starting_rotation()
    state = imu.State.from_dict(
        {
            "oriw": orientation.a,
            "orix": orientation.b,
            "oriy": orientation.c,
            "oriz": orientation.d,
        }
    )
    last_time = None

    TOL = 10.0

    # Stationary Data from a rocket launch pre-ignition
    for idx, (time_ns, control) in enumerate(stream_preignition()):
        if last_time is not None:
            dt = (time_ns - last_time) * 1e-9
            print("timekeeper", idx, dt)
        last_time = time_ns

        control = imu.Control.from_dict(control)
        state = imu.model(dt, state, control)
        assert state is not None

        expected_state = imu.State.from_dict(
            {
                r"\ddot{x}_{A}_{1}": 0.0,
                r"\ddot{x}_{A}_{2}": 0.0,
                r"\ddot{x}_{A}_{3}": 0.0,
                r"\dot{\phi}": 0.0,
                r"\dot{\psi}": 0.0,
                r"\dot{\theta}": 0.0,
                r"\dot{x}_{A}_{1}": 0.0,
                r"\dot{x}_{A}_{2}": 0.0,
                r"\dot{x}_{A}_{3}": 0.0,
                "oriw": orientation.a,
                "orix": orientation.b,
                "oriy": orientation.c,
                "oriz": orientation.d,
                r"x_{A}_{1}": 0.0,
                r"x_{A}_{2}": 0.0,
                r"x_{A}_{3}": 0.0,
            }
        )

        if not np.allclose(state.data, expected_state.data, atol=TOL):
            state.render_diff(expected_state)
            reference_time_zero_ns = 1602596210210000000
            print(
                "Time in ns:",
                time_ns,
                "Reference Zero in ns:",
                reference_time_zero_ns,
                "Delta (sec)",
                (reference_time_zero_ns - time_ns) * 1e-9,
            )
            1 / 0
        assert np.allclose(state.data, expected_state.data, atol=TOL)
