import sys
from os.path import basename

from formak.ui import *
from formak import cpp


def model_definition():
    dt = Symbol("dt")

    tp = trajectory_properties = {
        k: Symbol(k)
        for k in [
            "pose_x",
            "pose_y",
            "pose_z",
            "roll",
            "pitch",
            "yaw",
            "vel_x",
            "vel_y",
            "vel_z",
            "roll_rate",
            "pitch_rate",
            "yaw_rate",
        ]
    }

    acc_x, acc_y, acc_z = symbols(["acc_x", "acc_y", "acc_z"])

    state = set(tp.values())
    control = {acc_x, acc_y, acc_z}

    state_model = {
        tp["pose_x"]: tp["pose_x"] + dt * tp["vel_x"],
        tp["pose_y"]: tp["pose_y"] + dt * tp["vel_y"],
        tp["pose_z"]: tp["pose_z"] + dt * tp["vel_z"],
        tp["vel_x"]: tp["vel_x"] + dt * acc_x,
        tp["vel_y"]: tp["vel_y"] + dt * acc_y,
        tp["vel_z"]: tp["vel_z"] + dt * acc_z,
    }

    model = Model(dt=dt, state=state, control=control, state_model=state_model)
    return model


def test_python_Model_simple():
    model = model_definition()

    python_implementation = python.compile(model)

    state_vector = np.array([[0.0, 0.0, 0.0, 0.0]]).transpose()
    control_vector = np.array([[0.0]])

    state_vector_next = python_implementation.model(0.1, state_vector, control_vector)
