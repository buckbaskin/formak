from math import asin, atan, atan2, cos, pi, sin, sqrt

import numpy as np
import sympy as sy

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
        representation="quaternion",
    ):
        euler_angles = {yaw, pitch, roll}
        quaternion = {w, x, y, z}
        direct_cosine_matrix = {True if matrix is not None else None}
        print("Constructor")
        print({"yaw": yaw, "pitch": pitch, "roll": roll})

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
        self.representation = representation

        if representation == "quaternion":
            self.quaternion = np.array([[w, x, y, z]]).transpose()
            if euler_angles != {None}:
                print("Constructor Quaternion From Euler")
                self.quaternion = self._quaternion_from_euler(
                    yaw=yaw, pitch=pitch, roll=roll
                )
            elif matrix is not None:
                self.quaternion = self._quaternion_from_matrix(matrix=matrix)
            print(
                "Constructor Quaternion",
                self.quaternion,
                "Valid?",
                self._quaternion_valid(self.quaternion),
            )
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

        z = cos(roll / 2) * cos(pitch / 2) * sin(yaw / 2) - sin(roll / 2) * sin(
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
        return quaternion.shape == (4, 1) and np.allclose(np.sum(quaternion**2), 1.0)

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
        """
        Ψψ Psi   Yaw
        Θθ Theta Pitch
        Φφ Phi   Roll
        """
        c11 = matrix[0, 0]
        c12 = matrix[0, 1]
        c13 = matrix[0, 2]
        c23 = matrix[1, 2]
        c33 = matrix[2, 2]

        # All Combinations...
        # c13 ... sin(pitch) {pitch}
        # c11 / c12 ... -1/tan(yaw) {yaw}
        #   c12 / c11 = -tan(yaw)
        # c23 / c33 ... -tan(roll) {roll}

        pitch = asin(c13)
        yaw = -atan(c12 / c11)
        roll = -atan(c23 / c33)

        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    def _euler_valid(self, ypr):
        return set(ypr.keys()) == {"yaw", "pitch", "roll"}

    def _matrix_from_quaternion(self, w, x, y, z):
        c11 = pow(w, 2) + pow(x, 2) - pow(y, 2) - pow(z, 2)
        c12 = 2 * (x * y - w * z)
        c13 = 2 * (x * z + w * y)
        c21 = 2 * (x * y + w * z)
        c22 = pow(w, 2) - pow(x, 2) + pow(y, 2) - pow(z, 2)
        c23 = 2 * (y * z - w * x)
        c31 = 2 * (x * z - w * y)
        c32 = 2 * (y * z + w * x)
        c33 = pow(w, 2) - pow(x, 2) - pow(y, 2) + pow(z, 2)
        return np.array([[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]])

    def _matrix_from_euler(self, yaw, pitch, roll):
        """
        Ψψ Psi   Yaw
        Θθ Theta Pitch
        Φφ Phi   Roll
        """
        yaw_part = np.array(
            [
                [cos(yaw), -sin(yaw), 0],
                [sin(yaw), cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        pitch_part = np.array(
            [
                [cos(pitch), 0, sin(pitch)],
                [0, 1, 0],
                [-sin(pitch), 0, cos(pitch)],
            ]
        )
        roll_part = np.array(
            [
                [1, 0, 0],
                [0, cos(roll), -sin(roll)],
                [0, sin(roll), cos(roll)],
            ]
        )
        combined = roll_part @ pitch_part @ yaw_part

        return combined

    def _matrix_valid(self, matrix):
        return matrix.shape == (3, 3)

    def as_euler(self):
        if self.representation == "quaternion":
            w = self.quaternion[0, 0]
            x = self.quaternion[1, 0]
            y = self.quaternion[2, 0]
            z = self.quaternion[3, 0]
            return self._euler_from_quaternion(w=w, x=x, y=y, z=z)
        elif self.representation == "matrix":
            return self._euler_from_matrix(matrix=self.matrix)
        else:  # representation == "euler"
            return self.euler

    def as_quaternion(self):
        if self.representation == "quaternion":
            return self.quaternion
        elif self.representation == "matrix":
            return self._quaternion_from_matrix(matrix=self.matrix)
        else:  # representation == "euler"
            return self._quaternion_from_euler(
                yaw=self.euler["yaw"],
                pitch=self.euler["pitch"],
                roll=self.euler["roll"],
            )

    def as_matrix(self):
        if self.representation == "quaternion":
            w = self.quaternion[0, 0]
            x = self.quaternion[1, 0]
            y = self.quaternion[2, 0]
            z = self.quaternion[3, 0]
            return self._matrix_from_quaternion(w=w, x=x, y=y, z=z)
        elif self.representation == "matrix":
            return self.matrix
        else:  # representation == "euler"
            return self._matrix_from_euler(
                yaw=self.euler["yaw"],
                pitch=self.euler["pitch"],
                roll=self.euler["roll"],
            )
