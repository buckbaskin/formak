import sys
from datetime import datetime
from functools import reduce
from os.path import basename

import numpy as np
from sympy import cos, sin
from sympy.solvers.solveset import nonlinsolve

from formak import python, ui


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
    return ui.Matrix(s), rotation(*s)


def rotation_rate(roll, pitch, yaw):
    return ui.Matrix([roll, pitch, yaw])


def named_rotation_rate(name):
    s = ui.symbols([f"{name}_roll_rate", f"{name}_pitch_rate", f"{name}_yaw_rate"])
    return ui.Matrix(s), rotation_rate(*s)


def translation(x, y, z):
    return ui.Matrix([[x], [y], [z]])


def named_translation(name):
    return translation(f"{name}_pos_x", f"{name}_pos_y", f"{name}_pos_z")


def velocity(x, y, z):
    return ui.Matrix([[x], [y], [z]])


def named_velocity(name):
    return velocity(f"{name}_vel_x", f"{name}_vel_y", f"{name}_vel_z")


def acceleration(x, y, z):
    return ui.Matrix([[x], [y], [z]])


def named_acceleration(name):
    return acceleration(f"{name}_acc_x", f"{name}_acc_y", f"{name}_acc_z")


def model_definition(*, debug=False):
    start_time = datetime.now()

    ## Define Model

    dt = ui.Symbol("dt")

    # CON: Center Of Navigation
    CON_position_in_global_frame = named_translation("CON_pos")
    orientation_states, CON_orientation_in_global_frame = named_rotation("CON_ori")

    if debug:
        print("CON_orientation_in_global_frame")
        print(CON_orientation_in_global_frame)

    # Note: Needs body-fixed / moving frame math
    CON_velocity_in_global_frame = named_velocity("CON")

    IMU_position_in_CON_frame = named_translation("IMU")
    IMU_orientation_states, IMU_orientation_in_CON_frame = named_rotation("IMU_ori")

    (
        reading_orientation_rate_states,
        IMU_orientation_rates_in_IMU_frame,
    ) = named_rotation_rate("IMU_reading")
    (
        orientation_rate_rate_states,
        IMU_orientation_rate_rates_in_IMU_frame,
    ) = named_rotation_rate("IMU_reading_rate")
    IMU_acceleration_in_IMU_frame = named_acceleration("IMU_reading")
    IMU_velocity_in_CON_frame = named_velocity("IMU_in_CON")
    IMU_acceleration_in_CON_frame = named_acceleration("IMU_in_CON")

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

    (
        IMU_orientation_rate_states,
        IMU_orientation_rates_in_CON_frame,
    ) = named_rotation_rate("IMU_in_CON")

    CON_orientation_rates_in_CON_frame = (
        IMU_orientation_rates_in_CON_frame
        + IMU_orientation_in_CON_frame * IMU_orientation_rates_in_IMU_frame
    )

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
    calibration = reduce(
        lambda x, y: x | y.free_symbols,
        [
            IMU_position_in_CON_frame,
            IMU_orientation_in_CON_frame,
        ],
        set(),
    )
    control = reduce(
        lambda x, y: x | y.free_symbols,
        [
            IMU_orientation_rates_in_IMU_frame,
            IMU_acceleration_in_IMU_frame,
        ],
        set(),
    )

    CON_velocity_in_CON_frame = (
        CON_orientation_in_global_frame.transpose() * CON_velocity_in_global_frame
    )

    next_position = CON_position_in_global_frame + (dt * CON_velocity_in_global_frame)
    next_orientation = orientation_states + dt * (
        IMU_orientation_in_CON_frame * reading_orientation_rate_states
    )
    assert len(next_orientation) == 3

    next_velocity = CON_velocity_in_global_frame + (
        dt * CON_acceleration_in_global_frame
    )

    def state_model_composer():
        for p in zip(
            sorted(CON_position_in_global_frame.free_symbols, key=lambda x: x.name),
            next_position,
        ):
            yield p
        # Note: I know this is incorrect, but first order it's ok
        for o in zip(orientation_states, next_orientation):
            yield o
        for v in zip(
            sorted(CON_velocity_in_global_frame.free_symbols, key=lambda x: x.name),
            next_velocity,
        ):
            yield v

    if debug:
        for k, v in state_model_composer():
            print(" - ", k, " : ", v)

    state_model = {k: v for k, v in state_model_composer()}
    assert len(state_model) == 9

    ## Simplifying Assumptions

    simplifications = []  # List[Tuple(Symbol, Symbol)]
    # Rigid Body Assumption
    simplifications += [(s, 0.0) for s in IMU_acceleration_in_CON_frame.free_symbols]
    simplifications += [(s, 0.0) for s in IMU_velocity_in_CON_frame.free_symbols]
    # Constant rotation rates for each update
    simplifications += [(s, 0.0) for s in orientation_rate_rate_states]

    state_model = {k: v.subs(simplifications) for k, v in state_model.items()}

    print(f"pre ui.Model: {datetime.now() - start_time}")
    model = ui.Model(
        dt=dt,
        state=state,
        calibration=calibration,
        control=control,
        state_model=state_model,
    )
    print(f"post ui.Model: {datetime.now() - start_time}")
    return {
        "model": model,
        "state": state,
        "calibration": calibration,
        "control": control,
    }
