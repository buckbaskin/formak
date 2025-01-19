import pytest

from backend_py.compile_ekf import compile_ekf
from backend_py.compile_model import compile_model
from formak import exceptions, python
from problemdefinition.model import Model as UiModel
from sympy import Symbol, symbols


def test_Model_creation_calibration_mismatch():
    dt = Symbol("dt")

    ui_model = UiModel(
        dt=dt,
        state=set(symbols(["x"])),
        control=set(),
        calibration=set(),
        state_model={Symbol("x"): "x + a + b"},
    )

    with pytest.raises(exceptions.ModelConstructionError):
        compile_model(
            ui_model,
            calibration_map={Symbol("a"): 0.0, Symbol("b"): 0.0},
            config={},
        )


def test_Model_creation_calibration():
    dt = Symbol("dt")

    ui_model = UiModel(
        dt=dt,
        state=set(symbols(["x"])),
        control=set(),
        calibration=set(symbols(["a", "b"])),
        state_model={Symbol("x"): "x + a + b"},
    )

    model = compile_model(
        ui_model,
        calibration_map={Symbol("a"): 0.0, Symbol("b"): 0.0},
        config={},
    )

    assert model.arglist == symbols(["dt", "x", "a", "b"])

    dt = 0.1

    state_vector = model.State(x=0.0)
    assert (model.model(dt=dt, state=state_vector).data.transpose() == [0.0]).all()

    model = compile_model(
        ui_model,
        calibration_map={Symbol("a"): 5.0, Symbol("b"): 0.5},
        config={},
    )
    state_vector = model.State(x=-1.0)
    assert (model.model(dt=dt, state=state_vector).data.transpose() == [4.5]).all()


def test_EKF_creation_calibration():
    dt = Symbol("dt")

    a, b, x, y = symbols(["a", "b", "x", "y"])

    ui_model = UiModel(
        dt=dt,
        state={x},
        control=set(),
        calibration={a, b},
        state_model={x: x + a + b},
    )

    ekf = compile_ekf(
        symbolic_model=ui_model,
        process_noise={},
        sensor_models={y: {y: x + b}},
        sensor_noises={y: {y: 1}},
        calibration_map={a: 0.0, b: 0.0},
        config={},
    )

    dt = 0.1
    state_covariance = ekf.Covariance(x=1)

    state_vector = ekf.State(x=0.0)
    assert (
        ekf.process_model(
            dt=dt, state=state_vector, covariance=state_covariance
        ).state.data.transpose()
        == [0.0]
    ).all()

    ekf = compile_ekf(
        ui_model,
        process_noise={},
        sensor_models={y: {y: x + b}},
        sensor_noises={y: {y: 1}},
        calibration_map={a: 5.0, b: 0.5},
        config={},
    )
    state_vector = ekf.State(x=-1.0)
    assert (
        ekf.process_model(
            dt=dt, state=state_vector, covariance=state_covariance
        ).state.data.transpose()
        == [4.5]
    ).all()
