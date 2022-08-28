import pytest

from formak import py, ui


def test_UI_simple():
    dt = ui.Symbol("dt")

    tp = trajectory_properties = {k: ui.Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = ui.Symbol("thrust")

    state = set(tp.values())
    control = set([thrust])

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    model = ui.Model(state=state, control=control, state_model=state_model)

    python_implementation = py.compile(model)
    assert isinstance(python_implementation, py.Model)

    python_ekf = py.ExtendedKalmanFilter(
        state_model=python_implementation, sensor_models={}
    )
    assert isinstance(python_ekf, py.ExtendedKalmanFilter)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_UI_simple())
