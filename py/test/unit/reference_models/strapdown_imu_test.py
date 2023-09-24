from math import cos, pi, radians, sin

import numpy as np
import pytest
from formak.reference_models import strapdown_imu

from formak import python


def render_diff(state, expected_state):
    def diff_source():
        for key, model, expected in zip(
            state._arglist, state.data, expected_state.data
        ):
            float(model)
            float(expected)
            if np.allclose([model], [expected]):
                continue
            yield key, model, expected, model - expected

    print("Key".ljust(30), "Model".ljust(20), "Expected".ljust(20))
    for key, model, expected, diff in sorted(
        list(diff_source()), key=lambda row: abs(row[-1]), reverse=True
    ):
        print(str(key).ljust(30), str(model).rjust(20), str(expected).rjust(20))


def test_stationary():
    print("state", sorted(list(strapdown_imu.state), key=lambda s: str(s)))
    print("control", sorted(list(strapdown_imu.control), key=lambda s: str(s)))

    print("state_model")
    for k in sorted(list(strapdown_imu.state_model.keys()), key=lambda k: str(k)):
        v = strapdown_imu.state_model[k]
        print("key", k, "value", v)
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={strapdown_imu.g: 9.81},
    )
    assert imu is not None
    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel

    rate = 100  # Hz
    dt = 1.0 / rate

    # Circular Motion on X, Y plane rotating around Z axis
    radius = 5.0
    yaw_rate = pi

    # trvel = r * \theta
    velocity = radius * yaw_rate
    specific_force = radius * yaw_rate * yaw_rate

    state = imu.State.from_dict(
        {
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"\dot{x}_{A}_{2}": 0.0,
            r"\theta": radians(90),
        }
    )

    control = imu.Control.from_dict({imu_accel[2]: -9.81})

    print("dt", dt)
    print("state 0", state, state.data)
    print("control", control, control.data)
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
            r"\phi": 0.0,
            r"\psi": 0.0,
            r"\theta": radians(90),
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"x_{A}_{3}": 0.0,
        }
    )

    print("Diff")
    render_diff(state=state, expected_state=expected_state)

    assert np.allclose(state.data, expected_state.data)
    1 / 0


@pytest.mark.skip(reason="Simplify Debugging")
def test_circular_motion_xy_plane():
    print("state", sorted(list(strapdown_imu.state), key=lambda s: str(s)))
    print("control", sorted(list(strapdown_imu.control), key=lambda s: str(s)))

    print("state_model")
    for k in sorted(list(strapdown_imu.state_model.keys()), key=lambda k: str(k)):
        v = strapdown_imu.state_model[k]
        print("key", k, "value", v)
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={strapdown_imu.g: 9.81},
    )
    assert imu is not None
    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel

    rate = 100  # Hz
    dt = 1.0 / rate

    # Circular Motion on X, Y plane rotating around Z axis
    radius = 5.0
    yaw_rate = pi

    # trvel = r * \theta
    velocity = radius * yaw_rate
    specific_force = radius * yaw_rate * yaw_rate

    state = imu.State.from_dict(
        {
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"\dot{x}_{A}_{2}": velocity,
            r"\theta": radians(90),
        }
    )

    control_args = {imu_gyro[2]: pi, imu_accel[1]: specific_force}
    control = imu.Control.from_dict(control_args)

    print("dt", dt)
    print("state 0", state, state.data)
    print("control", control, control.data)
    state = imu.model(dt, state, control)
    assert state is not None

    expected_yaw = radians(90) + yaw_rate * dt
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
            r"x_{A}_{1}": radius * cos(expected_yaw - radians(90)),
            r"x_{A}_{2}": radius * sin(expected_yaw - radians(90)),
            r"x_{A}_{3}": 0.0,
        }
    )

    print("Diff")
    render_diff(state=state, expected_state=expected_state)

    assert np.allclose(state.data, expected_state.data)
    1 / 0
