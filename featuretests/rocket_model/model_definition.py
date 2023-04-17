import sys
from os.path import basename

from functools import reduce
from itertools import repeat
from formak import ui, python
import numpy as np
from sympy import sin, cos
from sympy.solvers.solveset import nonlinsolve


def rotation(roll, pitch, yaw):
    m_roll = ui.Matrix(
        [[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]]
    )
    m_pitch = ui.Matrix(
        [[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]]
    )
    m_yaw = ui.Matrix([[cos(yaw), sin(yaw), 0], [-sin(yaw), cos(yaw), 0], [0, 0, 1]])

    return m_roll * m_pitch * m_yaw


def named_rotation(name):
    s = ui.symbols([f"{name}_roll", f"{name}_pitch", f"{name}_yaw"])
    return ui.Matrix(s).transpose(), rotation(*s)


def translation(x, y, z):
    return ui.Matrix([[x], [y], [z]])


def named_translation(name):
    return translation(f"{name}_x", f"{name}_y", f"{name}_z")


def model_definition():
    dt = ui.Symbol("dt")

    # CON: Center Of Navigation
    CON_position_in_global_frame = named_translation("CON_pos")
    orientation_states, CON_orientation_in_global_frame = named_rotation("CON_ori")
    print("CON_orientation_in_global_frame")
    print(CON_orientation_in_global_frame)

    # Note: Needs body-fixed / moving frame math
    CON_velocity_in_global_frame = named_translation("CON_vel")

    IMU_position_in_CON_frame = named_translation("IMU_pos")
    IMU_orientation_states, IMU_orientation_in_CON_frame = named_rotation("IMU_ori")

    orientation_rate_states, IMU_orientation_rates_in_IMU_frame = named_rotation(
        "IMU_orate"
    )
    (
        orientation_rate_rate_states,
        IMU_orientation_rate_rates_in_IMU_frame,
    ) = named_rotation("IMU_orate_rate")
    IMU_acceleration_in_IMU_frame = named_translation("IMU_acc")

    CON_acceleration_in_CON_frame = (
        # a (measured by sensor, rotated to align with CON coordinates)
        IMU_orientation_in_CON_frame * IMU_acceleration_in_IMU_frame
        # tangential acceleration
        - IMU_orientation_rate_rates_in_IMU_frame.cross(IMU_position_in_CON_frame)
        # centripetal acceleration
        - IMU_orientation_rates_in_IMU_frame.cross(
            IMU_orientation_rates_in_IMU_frame.cross(IMU_position_in_CON_frame)
        )
        # acceleration relative to observer in CON frame
        - IMU_acceleration_in_CON_frame
        # Coriolis acceleration
        - (2.0 * IMU_orientation_rates_in_IMU_frame.cross(IMU_velocity_in_CON_frame))
    )

    # Note: Needs body-fixed / moving frame math
    # orientation_rate_states, CON_orientation_rates_in_CON_frame = named_rotation(
    #     "CON_orate"
    # )
    # CON_acceleration_in_CON_frame = named_translation("CON_acc")

    CON_orientation_rates_in_global_frame = (
        CON_orientation_in_global_frame * CON_orientation_rates_in_CON_frame
    )
    CON_acceleration_in_global_frame = (
        CON_orientation_in_global_frame * CON_acceleration_in_CON_frame
    )

    state = reduce(
        lambda x, y: x | y.free_symbols,
        [
            CON_position_in_global_frame,
            CON_orientation_in_global_frame,
            CON_velocity_in_global_frame,
        ],
        set(),
    )
    control = reduce(
        lambda x, y: x | y.free_symbols,
        [
            CON_orientation_rates_in_CON_frame,
            CON_acceleration_in_CON_frame,
        ],
        set(),
    )

    CON_velocity_in_CON_frame = (
        CON_orientation_in_global_frame.transpose() * CON_velocity_in_global_frame
    )

    next_position = CON_position_in_global_frame + (dt * CON_velocity_in_global_frame)
    # next_orientation = (
    #     dt * CON_orientation_rates_in_global_frame
    # ) * CON_orientation_in_global_frame
    next_velocity = CON_velocity_in_global_frame + (
        dt * CON_acceleration_in_global_frame
    )

    def state_model_composer():
        for p in zip(CON_position_in_global_frame.free_symbols, next_position):
            yield p
        # Note: I know this is incorrect, but first order it's ok
        for o in zip(
            orientation_states, orientation_states + dt * orientation_rate_states
        ):
            yield o
        for v in zip(CON_velocity_in_global_frame.free_symbols, next_velocity):
            yield v

    print("state model")
    for k, v in state_model_composer():
        print(" - ", k, " : ", v)

    state_model = {k: v for k, v in state_model_composer()}
    assert len(state_model) == 9

    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)
    return model
