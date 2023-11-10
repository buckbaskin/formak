import numpy as np
import sympy as sy

yaw, pitch, roll = sy.symbols(["yaw", "pitch", "roll"])
yaw_part = np.array(
    [
        [sy.cos(yaw), -sy.sin(yaw), 0],
        [sy.sin(yaw), sy.cos(yaw), 0],
        [0, 0, 1],
    ]
)
pitch_part = np.array(
    [
        [sy.cos(pitch), 0, sy.sin(pitch)],
        [0, 1, 0],
        [-sy.sin(pitch), 0, sy.cos(pitch)],
    ]
)
roll_part = np.array(
    [
        [1, 0, 0],
        [0, sy.cos(roll), -sy.sin(roll)],
        [0, sy.sin(roll), sy.cos(roll)],
    ]
)
sympy_mat = roll_part @ pitch_part @ yaw_part
print("Sympy Version of Mat from Euler")
# print(sympy_mat)


def piecewise():
    for i in range(3):
        for j in range(3):
            name = sy.Symbol(f"c{i+1}{j+1}")
            value = sympy_mat[i, j]
            if value.func != sy.core.add.Add:
                yield name, value


print("\n\nAll Combinations...")
for name, val in piecewise():
    if len(val.free_symbols) == 1:
        print(name, "...", val, val.free_symbols)

for namel, vall in piecewise():
    for namer, valr in piecewise():
        if namel == namer:
            continue
        if str(namel) > str(namer):
            continue
        combined = sy.simplify(vall / valr)

        if len(combined.free_symbols) == 1:
            print(namel, "/", namer, "...", combined, combined.free_symbols)
