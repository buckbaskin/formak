"""
Reference Model IMU

This demonstrates an example of "playing" data from a NASA launch through the
IMU model to generate a motion trajectory in the global frame.

Data Sample Source
https://data.nasa.gov/Aerospace/Deorbit-Descent-and-Landing-Flight-1-DDL-F1-/vicw-ivgd
"""
import csv
import os
from math import cos, hypot, radians, sin

import numpy as np
from formak.reference_models import strapdown_imu

from formak import python


def _line_to_control(line):
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

    return time_ns, (
        accel_1_ms2,
        accel_2_ms2,
        accel_3_ms2,
        body_rate_1,
        body_rate_2,
        body_rate_3,
    )


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

            print("per row gen", time_ns, "ns", ",".join(str(v) for v in parsed_row))
            yield time_ns, parsed_row


def test_example_usage_of_reference_model():
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={strapdown_imu.g: 9.81},
    )
    assert imu is not None
    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel

    rate = 50  # Hz
    dt = 1.0 / rate

    state = imu.State()
    last_time = None

    # Stationary Data from a rocket launch pre-ignition
    for idx, (time_ns, control) in enumerate(stream_control()):
        if last_time is not None:
            dt = time_ns - last_time
            print("timekeeper", idx, dt, dt * 1e-9)
        last_time = time_ns

        state = imu.model(dt, state, control)
        assert state is not None

        velocity = 0.0
        expected_yaw = 0.0
        yaw_rate = 0.0

        expected_state = imu.State.from_dict(
            {
                r"\ddot{x}_{A}_{1}": 0.0,
                r"\ddot{x}_{A}_{2}": 0.0,
                r"\ddot{x}_{A}_{3}": 0.0,
                r"\dot{\phi}": 0.0,
                r"\dot{\psi}": 0.0,
                r"\dot{\theta}": yaw_rate,
                r"\dot{x}_{A}_{1}": velocity * -sin(expected_yaw),
                r"\dot{x}_{A}_{2}": velocity * cos(expected_yaw),
                r"\dot{x}_{A}_{3}": 0.0,
                r"\phi": 0.0,
                r"\psi": 0.0,
                r"\theta": expected_yaw,
                r"x_{A}_{1}": cos(expected_yaw - radians(90)),
                r"x_{A}_{2}": sin(expected_yaw - radians(90)),
                r"x_{A}_{3}": 0.0,
            }
        )

        assert np.allclose(state.data, expected_state.data)
        1 / 0
