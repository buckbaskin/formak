import pytest
from formak.rotation import Rotation
from sympy import Matrix, Symbol, simplify

REPRESENTATIONS = ["quaternion", "matrix", "euler"]


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_symbolic_computation_euler(representation):
    reference = {k: Symbol(k) for k in ["yaw", "pitch", "roll"]}

    r = Rotation(**reference, representation=representation)

    print("Q")
    print(r.as_quaternion())

    print("M")
    print(r.as_matrix())

    print("E")
    print(r.as_euler())


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_symbolic_computation_quaternion(representation):
    reference = {k: Symbol(k) for k in ["w", "x", "y", "z"]}

    r = Rotation(**reference, representation=representation)

    print("Q")
    print(r.as_quaternion())

    print("M")
    print(r.as_matrix())

    print("E")
    print(r.as_euler())


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_symbolic_computation_matrix(representation):
    def matrix():
        for i in range(3):
            yield [f"c{i}{j}" for j in range(3)]

    r = Rotation(matrix=Matrix(list(matrix())), representation=representation)

    print("Q")
    print(r.as_quaternion())

    print("M")
    print(r.as_matrix())

    print("E")
    print(r.as_euler())
