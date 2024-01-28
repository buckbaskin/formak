from math import cos, pi, radians, sin

import numpy as np
from formak.reference_models import strapdown_imu
from matplotlib import pyplot as plt
from sympy import Quaternion

from formak import python, ui
from formak.common import plot_pair, plot_quaternion_timeseries


def test_stationary():
    print("state", sorted(list(strapdown_imu.state), key=lambda s: str(s)))
    print("control", sorted(list(strapdown_imu.control), key=lambda s: str(s)))

    print("state_model")
    for k in sorted(list(strapdown_imu.state_model.keys()), key=lambda k: str(k)):
        v = strapdown_imu.state_model[k]
        print("key", k, "value", v)
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={
            strapdown_imu.g: -9.81,
            strapdown_imu.coriw: 1.0,
            strapdown_imu.corix: 0.0,
            strapdown_imu.coriy: 0.0,
            strapdown_imu.coriz: 0.0,
            strapdown_imu.accel_sensor_bias[0]: 0.0,
            strapdown_imu.accel_sensor_bias[1]: 0.0,
            strapdown_imu.accel_sensor_bias[2]: 0.0,
        },
    )
    assert imu is not None
    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel

    rate = 100  # Hz
    dt = 1.0 / rate

    # Circular Motion on X, Y plane rotating around Z axis
    radius = 5.0
    yaw_rate = 0.0

    # travel = r * \psi
    velocity = 0.0
    specific_force = 0.0

    orientation = Quaternion.from_euler([radians(90), 0.0, 0.0], "zyx")

    state = imu.State.from_dict(
        {
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"\dot{x}_{A}_{2}": 0.0,
            "oriw": orientation.a,
            "orix": orientation.b,
            "oriy": orientation.c,
            "oriz": orientation.d,
        }
    )

    control = imu.Control.from_dict({imu_accel[2]: -9.81})

    print("dt", dt)
    print("state 0", state, state._arglist, "\n", state.data)
    print("control", control, control._arglist, "\n", control.data)
    state = imu.model(dt, state, control)
    assert state is not None

    orientation = Quaternion.from_euler([radians(90), 0.0, 0.0], "zyx")

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
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"x_{A}_{3}": 0.0,
        }
    )

    print("Diff")
    state.render_diff(expected_state)

    assert np.allclose(state.data, expected_state.data)


def test_circular_motion_xy_plane():
    print("state", sorted(list(strapdown_imu.state), key=lambda s: str(s)))
    print("control", sorted(list(strapdown_imu.control), key=lambda s: str(s)))

    print("state_model")
    for k in sorted(list(strapdown_imu.state_model.keys()), key=lambda k: str(k)):
        v = strapdown_imu.state_model[k]
        print("key", k, "value", v)
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={
            strapdown_imu.g: -9.81,
            strapdown_imu.coriw: 1.0,
            strapdown_imu.corix: 0.0,
            strapdown_imu.coriy: 0.0,
            strapdown_imu.coriz: 0.0,
            strapdown_imu.accel_sensor_bias[0]: 0.0,
            strapdown_imu.accel_sensor_bias[1]: 0.0,
            strapdown_imu.accel_sensor_bias[2]: 0.0,
        },
    )
    assert imu is not None
    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel

    rate = 100  # Hz
    dt = 1.0 / rate

    # Circular Motion on X, Y plane rotating around Z axis
    radius = 5.0
    yaw_rate = pi

    # travel = r * \psi
    velocity = radius * yaw_rate
    specific_force = radius * yaw_rate * yaw_rate

    orientation = Quaternion.from_euler([radians(90), 0.0, 0.0], "zyx")
    state = imu.State.from_dict(
        {
            r"\ddot{x}_{A}_{1}": -specific_force,
            r"\dot{\psi}": yaw_rate,
            r"\dot{x}_{A}_{2}": velocity,
            "oriw": orientation.a,
            "orix": orientation.b,
            "oriy": orientation.c,
            "oriz": orientation.d,
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
        }
    )

    control_args = {imu_gyro[2]: pi, imu_accel[1]: specific_force, imu_accel[2]: -9.81}
    control = imu.Control.from_dict(control_args)

    print("dt", dt)
    print("state 0", state, state._arglist, "\n", state.data)
    print("control", control, control._arglist, "\n", control.data)

    times = [0.0]
    states = [state.data]
    expected_states = [state.data]

    ALLOWED_TOL = 0.5

    # Check Initial State
    state = imu.model(0.0, state, control)
    expected_yaw = radians(90)
    expected_orientation = Quaternion.from_axis_angle([0.0, 0.0, 1.0], expected_yaw)
    expected_state = imu.State.from_dict(
        {
            r"\ddot{x}_{A}_{1}": -specific_force * sin(expected_yaw),
            r"\ddot{x}_{A}_{2}": specific_force * cos(expected_yaw),
            r"\ddot{x}_{A}_{3}": 0.0,
            r"\dot{\phi}": 0.0,
            r"\dot{\psi}": yaw_rate,
            r"\dot{\theta}": 0.0,
            r"\dot{x}_{A}_{1}": velocity * cos(expected_yaw),
            r"\dot{x}_{A}_{2}": velocity * sin(expected_yaw),
            r"\dot{x}_{A}_{3}": 0.0,
            "oriw": expected_orientation.a,
            "orix": expected_orientation.b,
            "oriy": expected_orientation.c,
            "oriz": expected_orientation.d,
            r"x_{A}_{1}": radius * sin(expected_yaw - yaw_rate * dt),
            r"x_{A}_{2}": radius * -cos(expected_yaw - yaw_rate * dt) + velocity * dt,
            r"x_{A}_{3}": 0.0,
        }
    )
    if not np.allclose(state.data, expected_state.data, atol=ALLOWED_TOL):
        print("Diff at index", 0)
        state.render_diff(expected_state)
        assert np.allclose(state.data, expected_state.data, atol=ALLOWED_TOL)

    break_idx = 3
    for idx in range(1, int(1.25 * rate)):
        # print("idx", idx)
        state = imu.model(dt, state, control)
        assert state is not None
        times.append(dt * idx)

        expected_yaw = radians(90) + yaw_rate * dt * idx
        expected_orientation = Quaternion.from_axis_angle([0.0, 0.0, 1.0], expected_yaw)
        expected_state = imu.State.from_dict(
            {
                r"\ddot{x}_{A}_{1}": -specific_force
                * sin(expected_yaw - yaw_rate * dt),
                r"\ddot{x}_{A}_{2}": specific_force * cos(expected_yaw - yaw_rate * dt),
                r"\ddot{x}_{A}_{3}": 0.0,
                r"\dot{\phi}": 0.0,
                r"\dot{\psi}": yaw_rate,
                r"\dot{\theta}": 0.0,
                r"\dot{x}_{A}_{1}": velocity * cos(expected_yaw),
                r"\dot{x}_{A}_{2}": velocity * sin(expected_yaw),
                r"\dot{x}_{A}_{3}": 0.0,
                "oriw": expected_orientation.a,
                "orix": expected_orientation.b,
                "oriy": expected_orientation.c,
                "oriz": expected_orientation.d,
                r"x_{A}_{1}": radius * sin(expected_yaw - yaw_rate * dt),
                r"x_{A}_{2}": radius * -cos(expected_yaw - yaw_rate * dt)
                + velocity * dt,
                r"x_{A}_{3}": 0.0,
            }
        )
        states.append(state.data)
        expected_states.append(expected_state.data)

        if not np.allclose(state.data, expected_state.data, atol=ALLOWED_TOL):
            print("Diff at index", idx, break_idx)
            state.render_diff(expected_state)
        else:
            break_idx = idx + 3

        if break_idx is not None and idx >= break_idx:
            break
    else:
        print("Diff at exit", idx, break_idx)
        state.render_diff(expected_state)

    states = np.array(states)
    expected_states = np.array(expected_states)

    print(state._arglist)
    plot_pair(
        states=states,
        expected_states=expected_states,
        arglist=state._arglist,
        x_name="x_{A}_{1}",
        y_name="x_{A}_{2}",
        file_id="pose_xy",
    )
    plot_pair(
        states=states,
        expected_states=expected_states,
        arglist=state._arglist,
        x_name=r"\dot{x}_{A}_{1}",
        y_name=r"\dot{x}_{A}_{2}",
        file_id="vel_xy",
    )
    plot_pair(
        states=states,
        expected_states=expected_states,
        arglist=state._arglist,
        x_name=r"\ddot{x}_{A}_{1}",
        y_name=r"\ddot{x}_{A}_{2}",
        file_id="accel_xy",
    )
    plot_quaternion_timeseries(
        times=times,
        states=states,
        expected_states=expected_states,
        arglist=state._arglist,
        x_name="ori",
        file_id="yaw_t",
    )
    print("Write image")

    assert np.allclose(state.data, expected_state.data, atol=ALLOWED_TOL)


def test_circular_motion_xz_plane():
    print("state", sorted(list(strapdown_imu.state), key=lambda s: str(s)))
    print("control", sorted(list(strapdown_imu.control), key=lambda s: str(s)))

    print("state_model")
    for k in sorted(list(strapdown_imu.state_model.keys()), key=lambda k: str(k)):
        v = strapdown_imu.state_model[k]
        print("key", k, "value", v)
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={
            strapdown_imu.g: -9.81,
            strapdown_imu.coriw: 1.0,
            strapdown_imu.corix: 0.0,
            strapdown_imu.coriy: 0.0,
            strapdown_imu.coriz: 0.0,
            strapdown_imu.accel_sensor_bias[0]: 0.0,
            strapdown_imu.accel_sensor_bias[1]: 0.0,
            strapdown_imu.accel_sensor_bias[2]: 0.0,
        },
    )
    assert imu is not None
    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel

    rate = 100  # Hz
    dt = 1.0 / rate

    # Circular Motion on X, Y plane rotating around Z axis
    radius = 5.0
    yaw_rate = pi

    # travel = r * \psi
    velocity = radius * yaw_rate
    specific_force = radius * yaw_rate * yaw_rate

    orientation = Quaternion.from_euler([0.0, -radians(90), 0.0], "zyx")
    state = imu.State.from_dict(
        {
            r"\ddot{x}_{A}_{1}": -specific_force,
            r"\dot{\psi}": yaw_rate,
            r"\dot{x}_{A}_{3}": velocity,
            "oriw": orientation.a,
            "orix": orientation.b,
            "oriy": orientation.c,
            "oriz": orientation.d,
            r"x_{A}_{1}": radius,
            r"x_{A}_{3}": 0.0,
        }
    )

    control_args = {
        imu_gyro[1]: -pi,
        imu_accel[0]: -9.81,
        imu_accel[2]: -specific_force,
    }
    control = imu.Control.from_dict(control_args)

    print("dt", dt)
    print("state 0", state, state._arglist, "\n", state.data)
    print("control", control, control._arglist, "\n", control.data)

    times = [0.0]
    states = [state.data]
    expected_states = [state.data]

    ALLOWED_TOL = 0.5

    # Check Initial State
    state = imu.model(0.0, state, control)
    expected_pitch = -radians(90)
    expected_orientation = Quaternion.from_axis_angle([0.0, 1.0, 0.0], expected_pitch)
    expected_state = imu.State.from_dict(
        {
            r"\ddot{x}_{A}_{1}": -specific_force * sin(expected_pitch),
            r"\ddot{x}_{A}_{2}": 0.0,
            r"\ddot{x}_{A}_{3}": specific_force * cos(expected_pitch),
            r"\dot{\phi}": 0.0,
            r"\dot{\psi}": 0.0,
            r"\dot{\theta}": -yaw_rate,
            r"\dot{x}_{A}_{1}": velocity * cos(expected_pitch),
            r"\dot{x}_{A}_{2}": 0.0,
            r"\dot{x}_{A}_{3}": -velocity * sin(expected_pitch),
            "oriw": expected_orientation.a,
            "orix": expected_orientation.b,
            "oriy": expected_orientation.c,
            "oriz": expected_orientation.d,
            r"x_{A}_{1}": radius * -sin(expected_pitch - yaw_rate * dt),
            r"x_{A}_{2}": 0.0,
            r"x_{A}_{3}": radius * -cos(expected_pitch - yaw_rate * dt) + velocity * dt,
        }
    )
    if not np.allclose(state.data, expected_state.data, atol=ALLOWED_TOL):
        print("Diff at index", 0)
        state.render_diff(expected_state)
        assert np.allclose(state.data, expected_state.data, atol=ALLOWED_TOL)

    # This could be extended to vary the gravity effect as the model rotates...
