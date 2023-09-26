import pytest

from formak import exceptions, python, ui


def test_Model_creation_calibration_mismatch():
    dt = ui.Symbol("dt")

    ui_model = ui.Model(
        dt=dt,
        state=set(ui.symbols(["x"])),
        control=set(),
        calibration=set(),
        state_model={ui.Symbol("x"): "x + a + b"},
    )

    with pytest.raises(exceptions.ModelConstructionError):
        python.compile(
            ui_model,
            calibration_map={ui.Symbol("a"): 0.0, ui.Symbol("b"): 0.0},
            config={},
        )


def test_Model_creation_calibration():
    dt = ui.Symbol("dt")

    ui_model = ui.Model(
        dt=dt,
        state=set(ui.symbols(["x"])),
        control=set(),
        calibration=set(ui.symbols(["a", "b"])),
        state_model={ui.Symbol("x"): "x + a + b"},
    )

    model = python.compile(
        ui_model,
        calibration_map={ui.Symbol("a"): 0.0, ui.Symbol("b"): 0.0},
        config={},
    )

    assert model.arglist == ui.symbols(["dt", "x", "a", "b"])

    dt = 0.1

    state_vector = model.State(x=0.0)
    assert (model.model(dt=dt, state=state_vector).data.transpose() == [0.0]).all()

    model = python.compile(
        ui_model,
        calibration_map={ui.Symbol("a"): 5.0, ui.Symbol("b"): 0.5},
        config={},
    )
    state_vector = model.State(x=-1.0)
    assert (model.model(dt=dt, state=state_vector).data.transpose() == [4.5]).all()


def test_EKF_creation_calibration():
    dt = ui.Symbol("dt")

    a, b, x, y = ui.symbols(["a", "b", "x", "y"])

    ui_model = ui.Model(
        dt=dt,
        state={x},
        control=set(),
        calibration={a, b},
        state_model={x: x + a + b},
    )

    ekf = python.compile_ekf(
        state_model=ui_model,
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

    ekf = python.compile_ekf(
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
