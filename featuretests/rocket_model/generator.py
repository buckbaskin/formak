from itertools import repeat

import numpy as np
from model_definition import (
    model_definition,
    named_acceleration,
    named_rotation_rate,
    named_translation,
)

from formak import cpp, ui

definition = model_definition()
ui_model = definition["model"]

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

(reading_orientation_rate_states, _) = named_rotation_rate("IMU_reading")
reading_acceleration_states = sorted(
    named_acceleration("IMU_reading").free_symbols, key=lambda x: x.name
)

process_noise = {
    k: v
    for k, v in list(zip(reading_orientation_rate_states, repeat(0.1)))
    + list(zip(reading_acceleration_states, repeat(1.0)))
}

CON_position_in_global_frame = named_translation("CON_pos")

cpp_implementation = cpp.compile_ekf(
    state_model=ui_model,
    process_noise={k: 1.0 for k in ui_model.control},
    sensor_models={
        "altitude": {ui.Symbol("altitude"): CON_position_in_global_frame[2]}
    },
    sensor_noises={"altitude": {"altitude": 1.0}},
    calibration_map=calibration_map,
    config=cpp.Config(common_subexpression_elimination=True),
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
