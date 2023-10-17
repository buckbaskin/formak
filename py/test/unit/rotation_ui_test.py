import numpy as np
import pytest
from formak.rotation import Rotation


# Note: Skipping Euler as a representation
@pytest.mark.parametrize("representation", ["quaternion", "matrix"])
def test_inverse(representation):
    reference = {"yaw": -0.2, "pitch": 0.1, "roll": -0.15}

    r = Rotation(**reference, representation=representation)

    inv = r.inverse()

    assert np.allclose(np.eye(3), inv.as_matrix() @ r.as_matrix())


@pytest.mark.parametrize("representation", ["quaternion", "matrix"])
def test_plus(representation):
    reference = {"yaw": -0.2, "pitch": 0.1, "roll": -0.15}

    r = Rotation(**reference, representation=representation)

    inv = r.inverse()

    assert np.allclose(np.eye(3), (inv + r).as_matrix())

    reference = {"yaw": 0.0, "pitch": -0.1, "roll": 0.0}

    r = Rotation(**reference, representation=representation)

    assert np.allclose(-0.2, (r + r).as_euler().pitch, atol=1e-2)


@pytest.mark.parametrize("representation", ["quaternion", "matrix"])
def test_minus(representation):
    reference = {"yaw": -0.2, "pitch": 0.1, "roll": -0.15}

    r = Rotation(**reference, representation=representation)

    assert np.allclose(np.eye(3), (r - r).as_matrix())

    reference = {"yaw": 0.0, "pitch": -0.1, "roll": 0.0}

    r2 = Rotation(**reference, representation=representation)

    assert np.allclose(0.2, (r - r2).as_euler().pitch, atol=1e-2)
