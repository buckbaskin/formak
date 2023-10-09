from math import pi

import numpy as np
import pytest
from formak.rotation import Rotation


def test_constructor():
    for r in ["quaternion", "matrix", "euler"]:
        Rotation(yaw=1.0, pitch=0.0, roll=0.0, representation=r)
        Rotation(w=1.0, x=0.0, y=0.0, z=0.0, representation=r)
        Rotation(matrix=np.eye(3), representation=r)


def test_euler_order():
    # TODO(buck): Should be yaw, pitch, roll, the 3-2-1 set of Euler angles
    r = Rotation(yaw=pi / 2, pitch=pi / 2, roll=0.0, representation="matrix")

    result = r.matrix
    expected = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    diff = result - expected
    print("result\n", result, "\nexpected\n", expected, "\ndiff\n", diff)
    assert np.allclose(result, expected)
