import numpy as np

from numpy.testing import assert_almost_equal
from formak import ui, python


def test_Model_creation_list():
    dt = ui.Symbol("dt")
    model = python.Model(ui.Model(dt, [], [], {}), {})

    assert model.arglist == ui.symbols(["dt"])


def test_Model_creation_set():
    dt = ui.Symbol("dt")
    model = python.Model(ui.Model(dt, set(), set(), {}), {})

    assert model.arglist == ui.symbols(["dt"])

    model = python.Model(
        ui.Model(dt, set(ui.symbols(["x"])), set(), {ui.Symbol("x"): "x"}), {}
    )

    assert model.arglist == ui.symbols(["dt", "x"])

    model = python.Model(
        ui.Model(
            dt,
            set(ui.symbols(["x", "y"])),
            set(),
            {ui.Symbol("x"): "x", ui.Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == ui.symbols(["dt", "x", "y"])

    model = python.Model(ui.Model(dt, set(), set(ui.symbols(["a"])), {}), {})

    assert model.arglist == ui.symbols(["dt", "a"])

    model = python.Model(
        ui.Model(
            dt, set(ui.symbols(["x"])), set(ui.symbols(["a"])), {ui.Symbol("x"): "x"}
        ),
        {},
    )

    assert model.arglist == ui.symbols(["dt", "x", "a"])

    model = python.Model(
        ui.Model(
            dt,
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x", ui.Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == ui.symbols(["dt", "x", "y", "a"])

    model = python.Model(ui.Model(dt, set(), set(ui.symbols(["a", "b"])), {}), {})

    assert model.arglist == ui.symbols(["dt", "a", "b"])

    model = python.Model(
        ui.Model(
            dt,
            set(ui.symbols(["x"])),
            set(ui.symbols(["a", "b"])),
            {ui.Symbol("x"): "x"},
        ),
        {},
    )

    assert model.arglist == ui.symbols(["dt", "x", "a", "b"])

    model = python.Model(
        ui.Model(
            dt,
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a", "b"])),
            {ui.Symbol("x"): "x", ui.Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == ui.symbols(["dt", "x", "y", "a", "b"])


def test_Model_impl_no_control():
    config = {}
    dt = 0.1

    model = python.Model(
        ui.Model(
            ui.Symbol("dt"),
            set(ui.symbols(["x", "y"])),
            set(),
            {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + 0.1"},
        ),
        config,
    )

    state_vector = np.array([[0.0, 0.0]]).transpose()
    assert (
        model.model(dt=dt, state_vector=state_vector).transpose() == [0.0, 0.1]
    ).all()

    state_vector = np.array([[0.0, 1.0]]).transpose()
    assert (
        model.model(dt=dt, state_vector=state_vector).transpose() == [0.0, 1.1]
    ).all()

    state_vector = np.array([[1.0, 0.0]]).transpose()
    assert (
        model.model(dt=dt, state_vector=state_vector).transpose() == [0.0, 0.1]
    ).all()

    state_vector = np.array([[1.0, 1.0]]).transpose()
    assert (
        model.model(dt=dt, state_vector=state_vector).transpose() == [1.0, 1.1]
    ).all()


def test_Model_impl_control():
    config = {}
    dt = 0.1

    model = python.Model(
        ui.Model(
            ui.Symbol("dt"),
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
        ),
        config,
    )

    control_vector = np.array([[0.2]])

    state_vector = np.array([[0.0, 0.0]]).transpose()
    assert_almost_equal(
        model.model(
            dt=dt, state_vector=state_vector, control_vector=control_vector
        ).transpose(),
        [[0.0, 0.02]],
    )

    state_vector = np.array([[0.0, 1.0]]).transpose()
    assert_almost_equal(
        model.model(
            dt=dt, state_vector=state_vector, control_vector=control_vector
        ).transpose(),
        [[0.0, 1.02]],
    )

    state_vector = np.array([[1.0, 0.0]]).transpose()
    assert_almost_equal(
        model.model(
            dt=dt, state_vector=state_vector, control_vector=control_vector
        ).transpose(),
        [[0.0, 0.02]],
    )

    state_vector = np.array([[1.0, 1.0]]).transpose()
    assert_almost_equal(
        model.model(
            dt=dt, state_vector=state_vector, control_vector=control_vector
        ).transpose(),
        [[1.0, 1.02]],
    )

