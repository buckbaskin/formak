from enum import Enum

from formak.ui import DesignManager, Model, Symbol


def test_model_simplification():
    dt = Symbol("dt")
    x = Symbol("x")
    model = Model(dt, {x}, set(), {x: x * x / x})

    assert model.state_model[x] == x


def test_ui_state_machine_ids_are_enums():
    initial_state = DesignManager(name="mercury")
    assert isinstance(initial_state.state_id(), Enum)

    assert all((isinstance(k, Enum) for k in initial_state.available_transitions()))
    assert all((isinstance(k, Enum) for k in initial_state.search("Fit Model")))
