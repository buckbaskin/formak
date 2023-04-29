import numpy as np

from formak import python, ui


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

    state_vector = np.array([[0.0]])
    assert (model.model(dt=dt, state=state_vector).transpose() == [0.0]).all()

    model = python.compile(
        ui_model,
        calibration_map={ui.Symbol("a"): 5.0, ui.Symbol("b"): 0.5},
        config={},
    )
    state_vector = np.array([[-1.0]])
    assert (model.model(dt=dt, state=state_vector).transpose() == [4.5]).all()


def test_EKF_creation_calibration():
    dt = ui.Symbol("dt")

    a, b, x, y = ui.symbols(["a", "b", "x", "y"])

    ui_model = ui.Model(
        dt=dt,
        state=set([x]),
        control=set(),
        calibration=set([a, b]),
        state_model={x: x + a + b},
    )

    ekf = python.compile_ekf(
        state_model=ui_model,
        process_noise={},
        sensor_models={y: {y: x + b}},
        sensor_noises={y: np.eye(1)},
        calibration_map={a: 0.0, b: 0.0},
        config={},
    )

    dt = 0.1
    state_covariance = np.eye(1)

    state_vector = np.array([[0.0]])
    assert (
        ekf.process_model(
            dt=dt, state=state_vector, covariance=state_covariance
        ).state.transpose()
        == [0.0]
    ).all()

    ekf = python.compile_ekf(
        ui_model,
        process_noise={},
        sensor_models={y: {y: x + b}},
        sensor_noises={y: np.eye(1)},
        calibration_map={a: 5.0, b: 0.5},
        config={},
    )
    state_vector = np.array([[-1.0]])
    assert (
        ekf.process_model(
            dt=dt, state=state_vector, covariance=state_covariance
        ).state.transpose()
        == [4.5]
    ).all()
