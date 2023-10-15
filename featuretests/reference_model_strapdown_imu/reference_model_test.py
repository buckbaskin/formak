"""
Reference Model IMU

This demonstrates an example of "playing" data from a NASA launch through the
IMU model to generate a motion trajectory in the global frame.

Data Sample Source
https://data.nasa.gov/Aerospace/Deorbit-Descent-and-Landing-Flight-1-DDL-F1-/vicw-ivgd
"""
import csv
import os
from math import asin, atan, cos, hypot, sin

import numpy as np
from formak.reference_models import strapdown_imu
from formak.rotation import Rotation

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


def stream_control():
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


def ypr_from_matrix(mat):
    c11 = mat[0, 0]
    c21 = mat[1, 0]
    c31 = mat[2, 0]
    c32 = mat[2, 1]
    c33 = mat[2, 2]

    roll = atan(c32 / c33)
    pitch = -asin(c31)
    yaw = atan(c21 / c11)
    return yaw, pitch, roll


def matrix_from_ypr(yaw, pitch, roll):
    yaw_mat = np.array(
        [
            [cos(1), -sin(1), 0],
            [sin(1), cos(1), 0],
            [0, 0, 1],
        ]
    )
    pitch_mat = np.array(
        [
            [cos(1), 0, sin(1)],
            [0, 1, 0],
            [-sin(1), 0, cos(1)],
        ]
    )
    roll_mat = np.array(
        [
            [1, 0, 0],
            [0, cos(1), -sin(1)],
            [0, sin(1), cos(1)],
        ]
    )
    mat = yaw_mat @ pitch_mat @ roll_mat
    mat = yaw_mat @ roll_mat @ pitch_mat
    # [y, p, r] -0.53962658
    # [y, r, p] 0.05619665
    # [p, y, r] -0.53962658
    # [p, r, y] -1.13544982
    # [r, p, y] -0.53962658
    # [r, y, p] -0.53962658
    return mat


# yaw
assert np.allclose(
    [1, 0, 0],
    ypr_from_matrix(
        np.array(
            [
                [cos(1), -sin(1), 0],
                [sin(1), cos(1), 0],
                [0, 0, 1],
            ]
        )
    ),
)
assert np.allclose(
    [-1, 0, 0],
    ypr_from_matrix(
        np.array(
            [
                [cos(-1), -sin(-1), 0],
                [sin(-1), cos(-1), 0],
                [0, 0, 1],
            ]
        )
    ),
)
# pitch
assert np.allclose(
    [0, 1, 0],
    ypr_from_matrix(
        np.array(
            [
                [cos(1), 0, sin(1)],
                [0, 1, 0],
                [-sin(1), 0, cos(1)],
            ]
        )
    ),
)
assert np.allclose(
    [0, -1, 0],
    ypr_from_matrix(
        np.array(
            [
                [cos(-1), 0, sin(-1)],
                [0, 1, 0],
                [-sin(-1), 0, cos(-1)],
            ]
        )
    ),
)
# roll
assert np.allclose(
    [0, 0, 1],
    ypr_from_matrix(
        np.array(
            [
                [1, 0, 0],
                [0, cos(1), -sin(1)],
                [0, sin(1), cos(1)],
            ]
        )
    ),
)
assert np.allclose(
    [0, 0, -1],
    ypr_from_matrix(
        np.array(
            [
                [1, 0, 0],
                [0, cos(-1), -sin(-1)],
                [0, sin(-1), cos(-1)],
            ]
        )
    ),
)


def starting_rotation():

    att_dcm_CON_IMU = np.array(
        [
            [-0.2477, -0.1673, 0.9543],
            [-0.0478, 0.9859, 0.1604],
            [-0.9677, -0.0059, -0.2522],
        ]
    )
    rotation = Rotation(matrix=att_dcm_CON_IMU)
    recreation = rotation.as_matrix()

    print("att")
    print(att_dcm_CON_IMU)
    print("rec")
    print(recreation)
    print("diff")
    print(att_dcm_CON_IMU - recreation)
    assert np.allclose(att_dcm_CON_IMU, recreation, atol=1e-4)

    return rotation


def test_example_usage_of_reference_model():
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={strapdown_imu.g: 9.81},
    )
    assert imu is not None

    rate = 50  # Hz
    dt = 1.0 / rate

    # yaw, pitch, roll = ui.symbols([r"\psi", r"\theta", r"\phi"])
    yaw, pitch, roll = starting_rotation().as_euler()
    state = imu.State.from_dict(
        {
            r"\phi": roll,  # roll
            r"\psi": yaw,  # yaw
            r"\theta": pitch,  # pitch
        }
    )
    last_time = None

    TOL = 0.12

    # Stationary Data from a rocket launch pre-ignition
    for idx, (time_ns, control) in enumerate(stream_control()):
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
                r"\phi": roll,  # roll
                r"\psi": yaw,  # yaw
                r"\theta": pitch,  # pitch
                r"x_{A}_{1}": 0.0,
                r"x_{A}_{2}": 0.0,
                r"x_{A}_{3}": 0.0,
            }
        )

        if not np.allclose(state.data, expected_state.data, atol=TOL):
            state.render_diff(expected_state)
            1 / 0
        assert np.allclose(state.data, expected_state.data, atol=TOL)
