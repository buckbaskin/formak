import pytest

from formak import python, ui


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

    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

    python_ekf = python.compile_ekf(
        state_model=model, sensor_models={}, config={"compile": True}
    )
    assert isinstance(python_ekf, python.ExtendedKalmanFilter)

    state_vector = [0.0, 0.0, 0.0, 0.0]
    control_vector = [0.0]

    state_vector_next = python_ekf.process_model(0.1, state_vector, control_vector)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_UI_simple())
