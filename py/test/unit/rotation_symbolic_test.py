import pytest
from sympy import Matrix, Quaternion, Symbol, simplify


def test_symbolic_computation_euler():
    reference = {k: Symbol(k) for k in ["yaw", "pitch", "roll"]}

    r = Quaternion(**reference)
    print(r.from_axis_angle.__doc__)

    print("Q")
    print(r)

    print("M")
    print(r.to_rotation_matrix())

    print("E")
    print(r.to_euler())


def test_symbolic_computation_quaternion():
    reference = {k: Symbol(k) for k in ["w", "x", "y", "z"]}

    r = Quaternion(*sorted(list(reference.keys())))

    print("Q")
    print(r)

    print("M")
    print(r.to_rotation_matrix())

    print("E")
    print(r.to_euler())


def test_symbolic_computation_matrix():
    def matrix():
        for i in range(3):
            yield [f"c{i}{j}" for j in range(3)]

    r = Quaternion.from_rotation_matrix(Matrix(list(matrix())))

    print("Q")
    print(r)

    print("M")
    print(r.to_rotation_matrix())

    print("E")
    print(r.to_euler())
