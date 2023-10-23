from sympy import Quaternion, integrate

from formak import ui

dt = ui.Symbol("dt")


def matrix_print(mat):
    rows, cols = mat.shape
    col_size = 80 // cols
    for row in range(rows):
        print(" ".join(str(col).ljust(col_size - 1) for col in mat[row, :]))


def axis_set(name):
    return ui.symbols([f"{name}_{{1}}", f"{name}_{{2}}", f"{name}_{{3}}"])


imu_gyro = axis_set(r"\omega")

oriw, orix, oriy, oriz = ui.symbols(["oriw", "orix", "oriy", "oriz"])
orientation = Quaternion(oriw, orix, oriy, oriz)
orientation_conjugate = Quaternion(oriw, -orix, -oriy, -oriz)

yaw_rate, pitch_rate, roll_rate = ui.symbols(
    [r"\dot{\psi}", r"\dot{\theta}", r"\dot{\phi}"]
)

# q * r * q_conj
_global_gyro_body_rates = orientation.mul(Quaternion(0, *imu_gyro)).mul(
    orientation_conjugate
)

imu_accel = axis_set("f")

global_pose = axis_set("x_{A}")
global_velocity = axis_set(r"\dot{x}_{A}")
global_accel = axis_set(r"\ddot{x}_{A}")
g = ui.Symbol("g")  # gravity

_accel_gravity = ui.Matrix([0, 0, -g])
assert _accel_gravity.shape == (3, 1)

_global_accel_body_rates = (
    orientation.mul(Quaternion(0, *imu_accel))
    .mul(orientation_conjugate)
    .add(Quaternion(0, *_accel_gravity))
)
_global_accel_body_rates = (
    orientation.to_rotation_matrix() * ui.Matrix(imu_accel) + _accel_gravity
)

_next_orientation = (0.5 * orientation.mul(Quaternion(0, *imu_gyro)) * dt).add(
    orientation
)

state = set(
    [oriw, orix, oriy, oriz, yaw_rate, pitch_rate, roll_rate]
    + global_pose
    + global_velocity
    + global_accel
)
control = set(imu_gyro + imu_accel)
calibration = {g}
state_model = {
    # Rotation
    yaw_rate: _global_gyro_body_rates.d,
    pitch_rate: _global_gyro_body_rates.c,
    roll_rate: _global_gyro_body_rates.b,
    # Rotation Integration
    oriw: _next_orientation.a,
    orix: _next_orientation.b,
    oriy: _next_orientation.c,
    oriz: _next_orientation.d,
    # Translation
    global_accel[0]: _global_accel_body_rates[0, 0],
    global_accel[1]: _global_accel_body_rates[1, 0],
    global_accel[2]: _global_accel_body_rates[2, 0],
    # Translation Integration
    global_velocity[0]: global_velocity[0]
    + integrate(_global_accel_body_rates[0, 0], dt),
    global_velocity[1]: global_velocity[1]
    + integrate(_global_accel_body_rates[1, 0], dt),
    global_velocity[2]: global_velocity[2]
    + integrate(_global_accel_body_rates[2, 0], dt),
    global_pose[0]: global_pose[0]
    + integrate(global_velocity[0] + integrate(_global_accel_body_rates[0, 0], dt), dt),
    global_pose[1]: global_pose[1]
    + integrate(global_velocity[1] + integrate(_global_accel_body_rates[1, 0], dt), dt),
    global_pose[2]: global_pose[2]
    + integrate(global_velocity[2] + integrate(_global_accel_body_rates[2, 0], dt), dt),
}

_process_noise = {
    imu_gyro[0]: 0.1,
    imu_gyro[1]: 0.1,
    imu_gyro[2]: 0.1,
    imu_accel[0]: 1.0,
    imu_accel[1]: 1.0,
    imu_accel[2]: 1.0,
}

symbolic_model = ui.Model(
    dt=dt,
    state=state,
    control=control,
    state_model=state_model,
    calibration=calibration,
)
