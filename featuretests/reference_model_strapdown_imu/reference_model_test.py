from formak.reference_models import strapdown_imu

from formak import python


def test_example_usage_of_reference_model():
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

    model = python.compile(
        symbolic_model=strapdown_imu.symbolic_model,
    )
    assert model is not None

    ekf = python.compile_ekf(
        state_model=strapdown_imu.symbolic_model,
        process_noise=process_noise,
        sensor_models={},
        sensor_noises={},
    )
    assert ekf is not None
