from itertools import permutations
from math import cos, pi, sin

import numpy as np
import pytest
from formak.rotation import Rotation

REPRESENTATIONS = ["quaternion", "matrix", "euler"]


def test_constructor():
    for r in REPRESENTATIONS:
        Rotation(yaw=1.0, pitch=0.0, roll=0.0, representation=r)
        Rotation(w=1.0, x=0.0, y=0.0, z=0.0, representation=r)
        Rotation(matrix=np.eye(3), representation=r)


def test_euler_order():
    """
    Ψψ Psi   Yaw
    Θθ Theta Pitch
    Φφ Phi   Roll
    """

    # TODO(buck): Should be yaw, pitch, roll, the 3-2-1 set of Euler angles
    reference = {"yaw": 1.0, "pitch": 1.0, "roll": 0.0}
    r = Rotation(**reference, representation="matrix")

    ypr = r.as_euler()
    for key in ["yaw", "pitch", "roll"]:
        if not np.allclose(ypr[key], reference[key]):
            print("as euler mismatch keys")
            print(key, "as_euler", ypr[key], "reference", reference[key])
        # assert np.allclose(ypr[key], reference[key])

    yaw_part = np.array(
        [
            [cos(reference["yaw"]), -sin(reference["yaw"]), 0],
            [sin(reference["yaw"]), cos(reference["yaw"]), 0],
            [0, 0, 1],
        ]
    )
    pitch_part = np.array(
        [
            [cos(reference["pitch"]), 0, sin(reference["pitch"])],
            [0, 1, 0],
            [-sin(reference["pitch"]), 0, cos(reference["pitch"])],
        ]
    )
    roll_part = np.array(
        [
            [1, 0, 0],
            [0, cos(reference["roll"]), -sin(reference["roll"])],
            [0, sin(reference["roll"]), cos(reference["roll"])],
        ]
    )
    print(
        "\nyaw\n",
        np.round(yaw_part, 8),
        "\npitch\n",
        np.round(pitch_part, 8),
        "\nroll\n",
        np.round(roll_part, 8),
    )

    mat_map = {"yaw": yaw_part, "pitch": pitch_part, "roll": roll_part}
    result = np.round(r.matrix, 8)
    for a, b, c in permutations(["yaw", "pitch", "roll"]):
        print("option", a, b, c)
        a = mat_map[a]
        b = mat_map[b]
        c = mat_map[c]
        option = a @ b @ c
        diff = np.round(result - option, 8)
        if np.allclose(option, result):
            print("Match!")

    expected = np.round(roll_part @ pitch_part @ yaw_part, 8)
    diff = np.round(result - expected, 8)
    print("result\n", result, "\nexpected\n", expected, "\ndiff\n", diff)
    assert np.allclose(result, expected)


def test_construct_to_output_consistency_euler():
    reference = {"yaw": 0.2, "pitch": -0.3, "roll": 0.4}
    arglist = sorted(list(reference.keys()))

    for representation in REPRESENTATIONS:
        r = Rotation(**reference, representation=representation)
        ypr = r.as_euler()

        try:
            assert np.allclose(
                [ypr[key] for key in arglist], [reference[key] for key in arglist]
            )
        except AssertionError:
            print(
                "Representation", representation, "Reference", reference, "Result", ypr
            )
            raise
        print("tried", representation)


def test_construct_to_output_consistency_quaternion():
    reference = np.array([[1, 0, 0, 0]]).transpose()
    w_ref = reference[0, 0]
    x_ref = reference[1, 0]
    y_ref = reference[2, 0]
    z_ref = reference[3, 0]

    for representation in REPRESENTATIONS:
        r = Rotation(w=w_ref, x=x_ref, y=y_ref, z=z_ref, representation=representation)
        quat = r.as_quaternion()

        try:
            assert np.allclose(reference, quat)
        except AssertionError:
            print(
                "Representation", representation, "Reference", reference, "Result", quat
            )
            raise
        print("tried", representation)


def test_construct_to_output_consistency_matrix():
    reference = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])

    for representation in REPRESENTATIONS:
        r = Rotation(matrix=reference, representation=representation)
        result = r.as_matrix()

        try:
            assert np.allclose(reference, result)
        except AssertionError:
            print(
                "Representation",
                representation,
                "Reference",
                reference,
                "Result",
                result,
            )
            raise
        print("tried", representation)
