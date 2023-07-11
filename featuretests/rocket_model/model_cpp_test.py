import numpy as np
from model_definition import model_definition

from formak import cpp, ui


def test_cpp_Model():
    definition = model_definition()
    model = definition["model"]

    calibration = {
        # orientation would need to invert a rotation matrix
        "IMU_ori_pitch": 0.0,
        "IMU_ori_roll": 0.0,
        "IMU_ori_yaw": 0.0,
        # pos_IMU_from_CON_in_CON     [m]     [-0.08035, 0.28390, -1.42333 ]
        "IMU_pos_x": -0.08035,
        "IMU_pos_y": 0.28390,
        "IMU_pos_z": -1.42333,
    }
    calibration_map = {ui.Symbol(k): v for k, v in calibration.items()}

    cpp.compile(
        model,
        calibration_map=calibration_map,
    )
