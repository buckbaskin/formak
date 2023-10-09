from math import asin, atan, atan2, cos, pi, sin, sqrt

import numpy as np

"""
Ψψ Psi   Yaw
Θθ Theta Pitch
Φφ Phi   Roll
"""


class Rotation:
    """
    Representation of rotations from one axis set to another

    Conversions follow the conventions defined in "Strapdown Inertial
    Navigation Technology" by Titterton and Weston (2nd edition)
    """

    def __init__(
        self,
        *,
        yaw=None,
        pitch=None,
        roll=None,
        w=None,
        x=None,
        y=None,
        z=None,
        matrix=None,
        representation="quaternion"
    ):
        euler_angles = {yaw, pitch, roll}
        quaternion = {w, x, y, z}
        direct_cosine_matrix = {True if matrix is not None else None}

        options = {
            "euler_angles": euler_angles,
            "quaternion": quaternion,
            "direct_cosine_matrix": direct_cosine_matrix,
        }
        populated_sets = {k: s for k, s in options.items() if s != {None}}
        populated_sets_count = len(populated_sets)
        if populated_sets_count != 1:
            error_str = (
                "Arguments provided for multiple rotation representations: "
                + ", ".join(sorted(list(populated_sets.keys())))
            )
            raise TypeError(error_str)

        allowed_representations = {"quaternion", "matrix", "euler"}
        if representation not in allowed_representations:
            raise ValueError(
                "representation={representation} must be replaced with one of the allowed strings: {allowed_representations}"
            )

        self.quaternion = None
        self.euler = None
        self.dcmatrix = None

        if representation == "quaternion":
            self.quaternion = np.array([[w, x, y, z]]).transpose()
            if euler_angles != {None}:
                self.quaternion = self._quaternion_from_euler(
                    yaw=yaw, pitch=pitch, roll=roll
                )
            elif matrix is not None:
                self.quaternion = self._quaternion_from_matrix(matrix=matrix)
            assert self._quaternion_valid(self.quaternion)
        elif representation == "matrix":
            self.matrix = matrix
            if quaternion != {None}:
                self.matrix = self._matrix_from_quaternion(w=w, x=x, y=y, z=z)
            elif euler_angles != {None}:
                self.matrix = self._matrix_from_euler(yaw=yaw, pitch=pitch, roll=roll)
            assert self._matrix_valid(self.matrix)
        else:  # representation == "euler"
            self.euler = {"yaw": yaw, "pitch": pitch, "roll": roll}
            if quaternion != {None}:
                self.euler = self._euler_from_quaternion(w=w, x=x, y=y, z=z)
            elif matrix is not None:
                self.euler = self._euler_from_matrix(matrix=matrix)
            self._euler_valid(self.euler)

    def _quaternion_from_euler(self, *, yaw, pitch, roll):
        w = cos(roll / 2) * cos(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * sin(
            pitch / 2
        ) * sin(yaw / 2)

        x = sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2) - cos(roll / 2) * sin(
            pitch / 2
        ) * sin(yaw / 2)

        y = cos(roll / 2) * sin(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * cos(
            pitch / 2
        ) * sin(yaw / 2)

        z = cos(roll / 2) * cos(pitch / 2) * sin(yaw / 2) + sin(roll / 2) * sin(
            pitch / 2
        ) * cos(yaw / 2)

        return np.array([[w, x, y, z]]).transpose()

    def _quaternion_from_matrix(self, *, matrix):
        # More detailed algorithm available in Shepperd [2]
        c11 = matrix[0, 0]
        c12 = matrix[0, 1]
        c13 = matrix[0, 2]
        c21 = matrix[1, 0]
        c22 = matrix[1, 1]
        c23 = matrix[1, 2]
        c31 = matrix[2, 0]
        c32 = matrix[2, 1]
        c33 = matrix[2, 2]
        w = 1 / 2 * pow(1 + c11 + c22 + c33, 1 / 2)
        x = 1 / (4 * w) * (c32 - c23)
        y = 1 / (4 * w) * (c13 - c31)
        z = 1 / (4 * w) * (c21 - c12)

        return np.array([[w, x, y, z]]).transpose()

    def _quaternion_valid(self, quaternion):
        return self.quaternion.shape == (4, 1)

    def _euler_from_quaternion(self, *, w, x, y, z):
        """
        "Quaternion to Euler angles (in 3-2-1 sequence) conversion"
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        """
        """
        Ψψ Psi   Yaw
        Θθ Theta Pitch
        Φφ Phi   Roll
        """
        roll = atan2(2 * (w * x + y * z), 1 - 2 * (pow(x, 2) + pow(y, 2)))
        pitch = -pi / 2 + 2 * atan2(
            sqrt(1 + 2 * (w * y - x * z)), sqrt(1 - 2 * (w * y - x * z))
        )
        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (pow(y, 2) + pow(z, 2)))
        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    def _euler_from_matrix(self, *, matrix):
        c11 = matrix[0, 0]
        c21 = matrix[1, 0]
        c31 = matrix[2, 0]
        c32 = matrix[2, 1]
        c33 = matrix[2, 2]

        roll = atan(c32 / c33)
        pitch = asin(-c31)
        yaw = atan(c21 / c11)

        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    def _euler_valid(self, ypr):
        return set(ypr.keys()) == {"yaw", "pitch", "roll"}

    def _matrix_from_quaternion(self, w, x, y, z):
        c11 = pow(w, 2) + pow(x, 2) - pow(y, 2) - pow(z, 2)
        c12 = 2 * (x * y - x * z)
        c13 = 2 * (x * z + x * y)
        c21 = 2 * (x * y + x * z)
        c22 = pow(x, 2) - pow(x, 2) + pow(y, 2) - pow(z, 2)
        c23 = 2 * (y * z - x * x)
        c31 = 2 * (x * z - x * y)
        c32 = 2 * (y * z + x * x)
        c33 = pow(x, 2) - pow(x, 2) - pow(y, 2) + pow(z, 2)
        return np.array([[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]])

    def _matrix_from_euler(self, yaw, pitch, roll):
        """
        Ψψ Psi   Yaw
        Θθ Theta Pitch
        Φφ Phi   Roll
        """
        c11 = cos(pitch) * cos(yaw)
        c12 = -cos(roll) * sin(yaw) + sin(roll) * sin(pitch) * cos(yaw)
        c13 = sin(roll) * sin(yaw) + cos(roll) * sin(pitch) * cos(yaw)
        c21 = cos(pitch) * sin(yaw)
        c22 = cos(roll) * cos(yaw) + sin(roll) * sin(pitch) * sin(yaw)
        c23 = -sin(roll) * cos(yaw) + cos(roll) * sin(pitch) * sin(yaw)
        c31 = -sin(pitch)
        c32 = sin(roll) * cos(pitch)
        c33 = cos(roll) * cos(pitch)
        return np.array([[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]])

    def _matrix_valid(self, matrix):
        return matrix.shape == (3, 3)
