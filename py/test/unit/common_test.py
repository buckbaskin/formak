import pytest

from formak.common import named_vector


def test_named_empty():
    Empty = named_vector("Empty", [])

    allowed = Empty()

    with pytest.raises(TypeError):
        disallowed = Empty(z=1.0)


def test_named_short():
    X = named_vector("X", ["x"])
    print(X)
    print(type(X))

    allowed = X(x=6)
    print(allowed)
    print(type(allowed))

    with pytest.raises(TypeError):
        disallowed = X(z=1.0)


def test_named_long():
    XYZ = named_vector("XYZ", ["x", "y", "z"])

    allowed = XYZ(x=6)
    allowed = XYZ(y=6)
    allowed = XYZ(y=6, z=5)

    with pytest.raises(TypeError):
        disallowed = XYZ(q=1.0)
