from sympy import (
    symbols,
    evaluate,
    expand,
    Symbol,
    factor,
    simplify,
    sin,
    cos,
    Quaternion,
)

i, j, k = symbols(["i", "j", "k"])
li, lj, lk = symbols(["li", "lj", "lk"])
ri, rj, rk = symbols(["ri", "rj", "rk"])

yaw, pitch, roll = symbols(["yaw", "pitch", "roll"])


def elements(prefix):
    return [
        l * r
        for l, r in zip(
            [1, i, j, k],
            symbols(
                [
                    f"{prefix}w",
                    f"{prefix}x",
                    f"{prefix}y",
                    f"{prefix}z",
                ]
            ),
        )
    ]


yaww, yawx, yawy, yawz = elements("yaw")
pitchw, pitchx, pitchy, pitchz = elements("pitch")
rollw, rollx, rolly, rollz = elements("roll")

print("yaw")
print(yaww, yawx, yawy, yawz)

left_subs = list(zip([i, j, k], [li, lj, lk]))
right_subs = list(zip([i, j, k], [ri, rj, rk]))
end_pairs_subs = [
    (li * ri, -1),
    (lj * rj, -1),
    (lk * rk, -1),
    (li * rj, k),
    (lj * ri, -k),
    (lj * rk, i),
    (lk * rj, -i),
    (lk * ri, j),
    (li * rk, -j),
]
end_single_subs = [(b, a) for (a, b) in left_subs] + [(b, a) for (a, b) in right_subs]

# with evaluate(False):
#     result =expand(
#             (pitchw + pitchx+ pitchy+ pitchz) *
#             (yaww + yawx+ yawy+ yawz)).subs(subs)
#     print('eval False')
#     print(result)


def multiply_quaternions(lhs, rhs):
    result = (
        expand(lhs.subs(left_subs) * rhs.subs(right_subs))
        .subs(end_pairs_subs)
        .subs(end_single_subs)
    )
    return result


pitch_times_yaw = multiply_quaternions(
    (pitchw + pitchx + pitchy + pitchz), (yaww + yawx + yawy + yawz)
)
print(pitch_times_yaw)


pitch_times_yaw = multiply_quaternions(
    multiply_quaternions(
        (cos(roll / 2) + sin(roll / 2) * i + 0 + 0),
        (cos(pitch / 2) + 0 + sin(pitch / 2) * j + 0),
    ),
    (cos(yaw / 2) + 0 + 0 + sin(yaw / 2) * k),
)
print(pitch_times_yaw)

print("\n\n=== Sympy Version ===\n")
w, x, y, z = symbols(["w", "x", "y", "z"])
rotation = Quaternion(w, x, y, z)
print(rotation)
print(list(filter(lambda s: s.startswith("from"), dir(rotation))))
print(list(filter(lambda s: s.startswith("to"), dir(rotation))))
print(list(dir(rotation)))
