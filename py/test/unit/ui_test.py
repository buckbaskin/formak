from enum import Enum
from typing import Dict

import numpy as np
import pytest
from formak.ui import DesignManager, Model, NisScore, Symbol
from formak.ui_state_machine import StateId

from formak import python


def test_model_simplification():
    dt = Symbol("dt")
    x = Symbol("x")
    model = Model(dt, {x}, set(), {x: x * x / x})

    assert model.state_model[x] == x


def test_ui_state_machine_ids_are_enums():
    initial_state = DesignManager(name="mercury")
    assert isinstance(initial_state.state_id(), Enum)
    assert initial_state.state_id() == StateId.Start

    # strings match function call names
    assert all(isinstance(k, str) for k in initial_state.available_transitions())
    assert all(isinstance(k, str) for k in initial_state.search(StateId.Fit_Model))


def test_fit_model_missing_parameter_defaults() -> None:
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

    initial_state = DesignManager(name="mercury")

    symbolic_model_state = initial_state.symbolic_model(model=symbolic_model)

    # Called with default parameters, empty data
    with pytest.raises(ValueError):
        fit_model_state = symbolic_model_state.fit_model(
            parameter_space={},
            data=[0, 0, 0],
        )


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
    adapter = python.SklearnEKFAdapter.Create(
        symbolic_model=symbolic_model,
        process_noise={thrust: 1.0},
        sensor_models={"simple": {tp["v"]: tp["v"]}},
        sensor_noises={"simple": {tp["v"]: 1.0}},
    )

    scoring_function = NisScore(estimator=adapter)

    assert scoring_function(X=np.ones((1, 2))) != 0.0
