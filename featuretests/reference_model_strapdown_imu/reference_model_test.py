from math import pi, radians

from formak.reference_models import strapdown_imu

from formak import python


def test_example_usage_of_reference_model():
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
    # velocity = radius * yaw_rate
    specific_force = radius * yaw_rate * yaw_rate

    state = imu.State.from_dict(
        {
            r"x_{A}_{1}": radius,
            r"x_{A}_{2}": 0.0,
            r"\theta": radians(90),
        }
    )

    control_args = {imu_gyro[2]: pi, imu_accel[1]: specific_force}
    control = imu.Control.from_dict(control_args)

    print("state 0", state)
    state = imu.model(dt, state, control)
    assert state is not None
    print("state 1", state)


def test_example_usage_of_reference_model_as_ekf():
    print("state", sorted(list(strapdown_imu.state), key=lambda s: str(s)))
    print("control", sorted(list(strapdown_imu.control), key=lambda s: str(s)))

    print("state_model")
    for k in sorted(list(strapdown_imu.state_model.keys()), key=lambda k: str(k)):
        v = strapdown_imu.state_model[k]
        print("key", k, "value", v)

    imu_gyro = strapdown_imu.imu_gyro
    imu_accel = strapdown_imu.imu_accel

    process_noise = {
        imu_gyro[0]: 0.1,
        imu_gyro[1]: 0.1,
        imu_gyro[2]: 0.1,
        imu_accel[0]: 1.0,
        imu_accel[1]: 1.0,
        imu_accel[2]: 1.0,
    }

    ekf = python.compile_ekf(
        state_model=strapdown_imu.symbolic_model,
        process_noise=process_noise,
        sensor_models={},
        sensor_noises={},
    )
    assert ekf is not None
