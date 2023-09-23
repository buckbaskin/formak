from sympy import Pow, cos, integrate, sec, simplify, sin, tan

from formak import python, ui

dt = ui.Symbol("dt")


def axis_set(name):
    return ui.symbols([f"{name}_{{1}}", f"{name}_{{2}}", f"{name}_{{3}}"])


imu_gyro = axis_set("\omega")

yaw, pitch, roll = ui.symbols([r"\psi", r"\theta", r"\phi"])
yaw_rate, pitch_rate, roll_rate = ui.symbols(
    [r"\dot{\psi}", r"\dot{\theta}", r"\dot{\phi}"]
)

gyro_rotations = ui.Matrix(
    [
        [0, sin(roll) * sec(pitch), cos(roll) * sec(pitch)],
        [0, cos(roll), -sin(roll)],
        [1, sin(roll) * tan(pitch), cos(roll) * tan(pitch)],
    ]
)

gyro_body_rates = gyro_rotations * ui.Matrix(imu_gyro)

assert gyro_body_rates.shape == (3, 1)

imu_accel = axis_set("f")

global_pose = axis_set("x_{A}")
global_velocity = axis_set("\dot{x}_{A}")
global_accel = axis_set("\ddot{x}_{A}")
g = ui.Symbol("g")  # gravity

accel_gravity = ui.Matrix([0, 0, -g])
assert accel_gravity.shape == (3, 1)

yaw_rotation = ui.Matrix(
    [
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1],
    ]
)
pitch_rotation = ui.Matrix(
    [
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)],
    ]
)
roll_rotation = ui.Matrix(
    [
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)],
    ]
)

accel_body_rates = (
    yaw_rotation * pitch_rotation * roll_rotation * ui.Matrix(imu_accel) + accel_gravity
)
assert accel_body_rates.shape == (3, 1)

state = set(
    [yaw, pitch, roll, yaw_rate, pitch_rate, roll_rate]
    + global_pose
    + global_velocity
    + global_accel
)
control = set(imu_gyro + imu_accel)
state_model = {
    # Rotation
    yaw_rate: gyro_body_rates[0, 0],
    pitch_rate: gyro_body_rates[1, 0],
    roll_rate: gyro_body_rates[2, 0],
    # Rotation Integration
    yaw: yaw + integrate(yaw_rate, dt),
    pitch: pitch + integrate(pitch_rate, dt),
    roll: roll + integrate(roll_rate, dt),
    # Translation
    global_accel[0]: accel_body_rates[0, 0],
    global_accel[1]: accel_body_rates[1, 0],
    global_accel[2]: accel_body_rates[2, 0],
    # Translation Integration
    global_velocity[0]: global_velocity[0] + integrate(global_accel[0], dt),
    global_velocity[1]: global_velocity[1] + integrate(global_accel[1], dt),
    global_velocity[2]: global_velocity[2] + integrate(global_accel[2], dt),
    global_pose[0]: global_pose[0]
    + integrate(global_velocity[0] + integrate(global_accel[0], dt), dt),
    global_pose[1]: global_pose[1]
    + integrate(global_velocity[1] + integrate(global_accel[1], dt), dt),
    global_pose[2]: global_pose[2]
    + integrate(global_velocity[2] + integrate(global_accel[2], dt), dt),
}

process_noise = {
    imu_gyro[0]: 0.1,
    imu_gyro[1]: 0.1,
    imu_gyro[2]: 0.1,
    imu_accel[0]: 1.0,
    imu_accel[1]: 1.0,
    imu_accel[2]: 1.0,
}


if __name__ == "__main__":
    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

    print("state", sorted(list(state), key=lambda s: str(s)))
    print("control", sorted(list(control), key=lambda s: str(s)))

    print("state_model")
    for k in sorted(list(state_model.keys()), key=lambda k: str(k)):
        v = state_model[k]
        print("key", k, "value", v)

    ekf = python.compile_ekf(
        state_model=model,
        process_noise=process_noise,
        sensor_models={},
        sensor_noises={},
    )
