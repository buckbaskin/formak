import pytest

from formak.common import named_covariance, named_vector


def test_named_vector_empty():
    Empty = named_vector("Empty", [])

    allowed = Empty()

    assert allowed.data.shape == (0, 1)

    with pytest.raises(TypeError):
        disallowed = Empty(z=1.0)


def test_named_vector_short():
    X = named_vector("X", ["x"])
    print(X)
    print(type(X))

    allowed = X(x=6)
    print(allowed)
    print(type(allowed))
    assert allowed.data.shape == (1, 1)

    with pytest.raises(TypeError):
        disallowed = X(z=1.0)


def test_named_vector_long():
    XYZ = named_vector("XYZ", ["x", "y", "z"])

    allowed = XYZ(x=6)
    allowed = XYZ(y=6)
    allowed = XYZ(y=6, z=5)

    assert allowed.data.shape == (3, 1)

    with pytest.raises(TypeError):
        disallowed = XYZ(q=1.0)


def test_named_covariance_empty():
    Empty = named_covariance("Empty", [])

    allowed = Empty()

    assert allowed.data.shape == (0, 0)

    with pytest.raises(TypeError):
        disallowed = Empty(z=1.0)


def test_named_covariance_short():
    X = named_covariance("X", ["x"])
    print(X)
    print(type(X))

    allowed = X(x=6)
    print(allowed)
    print(type(allowed))

    assert allowed.data.shape == (1, 1)

    with pytest.raises(TypeError):
        disallowed = X(z=1.0)


def test_named_covariance_long():
    XYZ = named_covariance("XYZ", ["x", "y", "z"])

    allowed = XYZ(x=6)
    allowed = XYZ(y=6)
    allowed = XYZ(y=6, z=5)

    assert allowed.data.shape == (3, 3)

    with pytest.raises(TypeError):
        disallowed = XYZ(q=1.0)
