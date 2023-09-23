from sympy import cos, integrate, sec, sin, tan

from formak import ui

dt = ui.Symbol("dt")


def axis_set(name):
    return ui.symbols([f"{name}_{{1}}", f"{name}_{{2}}", f"{name}_{{3}}"])


imu_gyro = axis_set(r"\omega")

yaw, pitch, roll = ui.symbols([r"\psi", r"\theta", r"\phi"])
yaw_rate, pitch_rate, roll_rate = ui.symbols(
    [r"\dot{\psi}", r"\dot{\theta}", r"\dot{\phi}"]
)

_gyro_rotations = ui.Matrix(
    [
        [0, sin(roll) * sec(pitch), cos(roll) * sec(pitch)],
        [0, cos(roll), -sin(roll)],
        [1, sin(roll) * tan(pitch), cos(roll) * tan(pitch)],
    ]
)

_gyro_body_rates = _gyro_rotations * ui.Matrix(imu_gyro)

assert _gyro_body_rates.shape == (3, 1)

imu_accel = axis_set("f")

global_pose = axis_set("x_{A}")
global_velocity = axis_set(r"\dot{x}_{A}")
global_accel = axis_set(r"\ddot{x}_{A}")
g = ui.Symbol("g")  # gravity

_accel_gravity = ui.Matrix([0, 0, -g])
assert _accel_gravity.shape == (3, 1)

_yaw_rotation = ui.Matrix(
    [
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1],
    ]
)
_pitch_rotation = ui.Matrix(
    [
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)],
    ]
)
_roll_rotation = ui.Matrix(
    [
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)],
    ]
)

_accel_body_rates = (
    _yaw_rotation * _pitch_rotation * _roll_rotation * ui.Matrix(imu_accel)
    + _accel_gravity
)
assert _accel_body_rates.shape == (3, 1)

state = set(
    [yaw, pitch, roll, yaw_rate, pitch_rate, roll_rate]
    + global_pose
    + global_velocity
    + global_accel
)
control = set(imu_gyro + imu_accel)
state_model = {
    # Rotation
    yaw_rate: _gyro_body_rates[0, 0],
    pitch_rate: _gyro_body_rates[1, 0],
    roll_rate: _gyro_body_rates[2, 0],
    # Rotation Integration
    yaw: yaw + integrate(yaw_rate, dt),
    pitch: pitch + integrate(pitch_rate, dt),
    roll: roll + integrate(roll_rate, dt),
    # Translation
    global_accel[0]: _accel_body_rates[0, 0],
    global_accel[1]: _accel_body_rates[1, 0],
    global_accel[2]: _accel_body_rates[2, 0],
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

_process_noise = {
    imu_gyro[0]: 0.1,
    imu_gyro[1]: 0.1,
    imu_gyro[2]: 0.1,
    imu_accel[0]: 1.0,
    imu_accel[1]: 1.0,
    imu_accel[2]: 1.0,
}

ui_model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)
