from sympy import cos, sec, sin, tan

from formak import ui

dt = ui.Symbol("dt")

imu_gyro = ui.symbols(["w_1", "w_2", "w_3"])
imu_accel = ui.symbols(["f_1", "f_2", "f_3"])

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

body_rates = gyro_rotations * ui.Matrix(imu_gyro)

state = set([yaw, pitch, roll, yaw_rate, pitch_rate, roll_rate])

assert body_rates.shape == (3, 1)

state_model = {
    yaw: yaw + dt * yaw_rate,
    yaw_rate: body_rates[0, 0],
    pitch: pitch + dt * pitch_rate,
    pitch_rate: body_rates[1, 0],
    roll: roll + dt * roll_rate,
    roll_rate: body_rates[2, 0],
}

if __name__ == "__main__":
    print(gyro_rotations.shape)
    print(gyro_rotations)
    print(ui.Matrix(imu_gyro).shape)
    print(ui.Matrix(imu_gyro))

    print("state_model")
    for k in sorted(list(state_model.keys()), key=lambda k: str(k)):
        v = state_model[k]
        print("key", k, "value", v)
    1 / 0
