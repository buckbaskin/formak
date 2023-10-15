from itertools import permutations, product
from math import cos, pi, sin

import numpy as np
import pytest
from formak.rotation import Rotation
from scipy.spatial.transform import Rotation as scipy_Rot
from sympy import Matrix, Symbol, simplify

REPRESENTATIONS = ["quaternion", "matrix", "euler"]


def test_constructor():
    for r in REPRESENTATIONS:
        Rotation(yaw=1.0, pitch=0.0, roll=0.0, representation=r)
        Rotation(w=1.0, x=0.0, y=0.0, z=0.0, representation=r)
        Rotation(matrix=np.eye(3), representation=r)


def YAW_PRINCIPALS():
    yield [1.0, 0.0, 0.0]
    yield [0.0, 1.0, 0.0]
    yield [0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "representation,principals",
    product(REPRESENTATIONS, YAW_PRINCIPALS()),
    ids=(
        f"{rep}, y{y} p{p} r{r}"
        for rep, (y, p, r) in product(REPRESENTATIONS, YAW_PRINCIPALS())
    ),
)
def test_principal_axis(representation, principals):
    def expected(*, yaw, pitch, roll):
        x, y, z, w = scipy_Rot.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        return np.array([[w, x, y, z]]).transpose()

    yaw, pitch, roll = principals

    print(representation, "Principle", yaw, pitch, roll)
    rotation = Rotation(yaw=yaw, pitch=pitch, roll=roll, representation=representation)

    quaternion = rotation.as_quaternion()
    print(
        "Test Result Quaternion",
        rotation._quaternion_valid(quaternion),
        "\n",
        quaternion,
    )
    expected_quaternion = expected(yaw=yaw, pitch=pitch, roll=roll)
    print(
        "Test Expected Quaternion",
        rotation._quaternion_valid(expected_quaternion),
        "\n",
        expected_quaternion,
    )

    assert rotation._quaternion_valid(quaternion)
    assert rotation._quaternion_valid(expected_quaternion)
    assert np.allclose(quaternion, expected_quaternion)

    euler = rotation.as_euler()

    assert np.allclose(euler.yaw, yaw)
    assert np.allclose(euler.pitch, pitch)
    assert np.allclose(euler.roll, roll)


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
    assert np.allclose(list(ypr), [reference[key] for key in ["yaw", "pitch", "roll"]])

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


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_construct_to_output_consistency_euler(representation):
    reference = {"yaw": 0.2, "pitch": -0.3, "roll": 0.4}
    arglist = ["yaw", "pitch", "roll"]

    def expected(*, yaw, pitch, roll):
        print("scipy_Rot from ", yaw, pitch, roll)
        r = scipy_Rot.from_euler("xyz", [roll, pitch, yaw])
        if representation == "quaternion":
            x, y, z, w = r.as_quat()
            print("Q\n", np.array([[w, x, y, z]]).transpose())

    expected(
        yaw=reference["yaw"],
        pitch=reference["pitch"],
        roll=reference["roll"],
    )

    r = Rotation(**reference, representation=representation)
    ypr = r.as_euler()

    try:
        assert np.allclose(list(ypr), [reference[key] for key in arglist])
    except AssertionError:
        print(
            "\nRepresentation\n",
            representation,
            "\nReference\n",
            reference,
            "\nResult\n",
            ypr,
        )
        raise


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_construct_to_output_consistency_quaternion(representation):
    reference = np.array([[1, 0, 0, 0]]).transpose()
    w_ref = reference[0, 0]
    x_ref = reference[1, 0]
    y_ref = reference[2, 0]
    z_ref = reference[3, 0]

    r = Rotation(w=w_ref, x=x_ref, y=y_ref, z=z_ref, representation=representation)
    quat = r.as_quaternion()

    try:
        assert np.allclose(reference, quat)
    except AssertionError:
        print(
            "\nRepresentation\n",
            representation,
            "\nReference\n",
            reference,
            "\nResult\n",
            quat,
        )
        raise
    print("tried", representation)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_construct_to_output_consistency_matrix(representation):
    reference = np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ]
    )
    assert np.allclose(1.0, np.linalg.det(reference))

    def expected():
        x, y, z, w = scipy_Rot.from_matrix(reference).as_quat()
        print("Expected Q\n", np.array([[w, x, y, z]]).transpose())

    expected()

    r = Rotation(matrix=reference, representation=representation)
    result = r.as_matrix()

    try:
        assert np.allclose(reference, result)
    except AssertionError:
        print(
            "\nRepresentation\n",
            representation,
            "\nReference\n",
            reference,
            "\nResult\n",
            result,
        )
        raise
    print("tried", representation)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_symbolic_computation_euler(representation):
    reference = {k: Symbol(k) for k in ["yaw", "pitch", "roll"]}

    r = Rotation(**reference, representation=representation)

    print("Q")
    print(simplify(r.as_quaternion()))

    print("M")
    print(simplify(r.as_matrix()))

    print("E")
    print(simplify(simplify(r.as_euler())))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_symbolic_computation_quaternion(representation):
    reference = {k: Symbol(k) for k in ["w", "x", "y", "z"]}

    r = Rotation(**reference, representation=representation)

    print("Q")
    print(simplify(r.as_quaternion()))

    print("M")
    print(simplify(r.as_matrix()))

    print("E")
    print(simplify(simplify(r.as_euler())))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_symbolic_computation_matrix(representation):
    def matrix():
        for i in range(3):
            yield [f"c{i}{j}" for j in range(3)]

    r = Rotation(matrix=Matrix(list(matrix())), representation=representation)

    print("Q")
    print(simplify(r.as_quaternion()))

    print("M")
    print(simplify(r.as_matrix()))

    print("E")
    print(simplify(simplify(r.as_euler())))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_inverse(representation):
    reference = {"yaw": -0.2, "pitch": 0.1, "roll": -0.15}

    r = Rotation(**reference, representation=representation)

    inv = r.inverse()

    assert np.allclose(np.eye(3), inv.as_matrix() @ r.as_matrix())
