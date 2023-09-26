from math import cos, pi, radians, sin

import numpy as np
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

    print("Key".ljust(30), "Model".ljust(15), "Expected".ljust(15), "Diff".ljust(15))
    for key, model, expected, diff in sorted(
        list(diff_source()), key=lambda row: abs(row[-1]), reverse=True
    ):
        print(
            str(key).ljust(30),
            str(model).rjust(15),
            str(expected).rjust(15),
            str(diff).rjust(15),
        )


def test_stationary():
    print("state", sorted(list(strapdown_imu.state), key=lambda s: str(s)))
    print("control", sorted(list(strapdown_imu.control), key=lambda s: str(s)))

    print("state_model")
    for k in sorted(list(strapdown_imu.state_model.keys()), key=lambda k: str(k)):
        v = strapdown_imu.state_model[k]
        print("key", k, "value", v)
    imu = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
        calibration_map={strapdown_imu.g: -9.81},
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

    state = imu.State.from_dict(
        {
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"\dot{x}_{A}_{2}": 0.0,
            r"\psi": radians(90),
        }
    )

    control = imu.Control.from_dict({imu_accel[2]: -9.81})

    print("dt", dt)
    print("state 0", state, state._arglist, "\n", state.data)
    print("control", control, control._arglist, "\n", control.data)
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
            r"\psi": radians(90),
            r"\theta": 0.0,
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"x_{A}_{3}": 0.0,
        }
    )

    print("Diff")
    render_diff(state=state, expected_state=expected_state)

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
        calibration_map={strapdown_imu.g: -9.81},
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

    state = imu.State.from_dict(
        {
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"\dot{x}_{A}_{2}": velocity,
            r"\psi": radians(90),
        }
    )

    control_args = {imu_gyro[2]: pi, imu_accel[1]: specific_force, imu_accel[2]: -9.81}
    control = imu.Control.from_dict(control_args)

    print("dt", dt)
    print("state 0", state, state._arglist, "\n", state.data)
    print("control", control, control._arglist, "\n", control.data)

    for idx in range(1, 2 * rate):
        state = imu.model(dt, state, control)
        assert state is not None

        expected_yaw = radians(90) + yaw_rate * dt * (idx - 1)
        expected_radius_angle = expected_yaw - radians(90)
        expected_state = imu.State.from_dict(
            {
                r"\ddot{x}_{A}_{1}": -specific_force * cos(expected_radius_angle),
                r"\ddot{x}_{A}_{2}": -specific_force * sin(expected_radius_angle),
                r"\ddot{x}_{A}_{3}": 0.0,
                r"\dot{\phi}": 0.0,
                r"\dot{\psi}": yaw_rate,
                r"\dot{\theta}": 0.0,
                r"\dot{x}_{A}_{1}": velocity * -cos(expected_yaw),
                r"\dot{x}_{A}_{2}": velocity * sin(expected_yaw),
                r"\dot{x}_{A}_{3}": 0.0,
                r"\phi": 0.0,
                r"\psi": expected_yaw,
                r"\theta": 0.0,
                r"x_{A}_{1}": radius * cos(expected_radius_angle),
                r"x_{A}_{2}": radius * sin(expected_radius_angle) + velocity * dt,
                r"x_{A}_{3}": 0.0,
            }
        )

        if not np.allclose(state.data, expected_state.data, atol=5e-3):
            print("Diff at index", idx)
            render_diff(state=state, expected_state=expected_state)

            assert np.allclose(state.data, expected_state.data, atol=5e-3)
    1 / 0
