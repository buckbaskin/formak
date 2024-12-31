"""
Feature Test.

Lay out a new model

Passes if the Model is constructed without exceptions
"""

from formak.problemdefinition.model import Model as UiModel
from sympy import Symbol


def test_UI_simple():
    dt = Symbol("dt")

    tp = _trajectory_properties = {k: Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = Symbol("thrust")

    state = set(tp.values())
    control = {thrust}

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    _model = UiModel(dt=dt, state=state, control=control, state_model=state_model)
