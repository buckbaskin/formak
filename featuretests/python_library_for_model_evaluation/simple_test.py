"""
Feature Test.

Create a Python model

Passes if the Python model runs
"""

from formak import python, ui


def test_python_Model_simple():
    dt = ui.Symbol("dt")

    tp = _trajectory_properties = {k: ui.Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = ui.Symbol("thrust")

    state = set(tp.values())
    control = {thrust}

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

    python_implementation = python.compile(model)

    state_vector = python_implementation.State()
    control_vector = python_implementation.Control()

    _state_vector_next = python_implementation.model(0.1, state_vector, control_vector)
