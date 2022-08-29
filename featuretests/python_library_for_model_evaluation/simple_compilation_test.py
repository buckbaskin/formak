import pytest

from formak.ui import *
from formak import python


def test_UI_simple():
    dt = Symbol("dt")

    tp = trajectory_properties = {k: Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = Symbol("thrust")

    state = set(tp.values())
    control = set([thrust])

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    model = Model(state=state, control=control, state_model=state_model)

    python_implementation = python.compile(model, {"compile": True})

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_UI_simple())
