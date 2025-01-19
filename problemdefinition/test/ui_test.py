import numpy as np

from backend_py.sklearn import SklearnEKFAdapter
from problemdefinition.config_view import ConfigView
from problemdefinition.model import Model
from problemdefinition.nis_score import NisScore
from sympy import Symbol


def test_model_simplification():
    dt = Symbol("dt")
    x = Symbol("x")
    model = Model(dt, {x}, set(), {x: x * x / x})

    assert model.state_model[x] == x


def test_non_zero_nis_score():
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

    symbolic_model = Model(dt=dt, state=state, control=control, state_model=state_model)
    adapter = SklearnEKFAdapter.Create(
        symbolic_model=symbolic_model,
        process_noise={thrust: 1.0},
        sensor_models={"simple": {tp["v"]: tp["v"]}},
        sensor_noises={"simple": {tp["v"]: 1.0}},
    )

    scoring_function = NisScore()

    assert scoring_function(estimator=adapter, X=np.ones((1, 2))) != 0.0


def test_ConfigView():
    params = {"common_subexpression_elimination": True}
    view = ConfigView(params)

    assert view.common_subexpression_elimination
