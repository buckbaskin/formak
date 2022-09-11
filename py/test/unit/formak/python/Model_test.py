import numpy as np
import warnings

from hypothesis import given, reject
from hypothesis.strategies import floats
from numpy.testing import assert_almost_equal
from formak import ui, python

warnings.filterwarnings("error")


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
    assert (model.model(dt=dt, state=state_vector).transpose() == [0.0, 0.1]).all()

    state_vector = np.array([[0.0, 1.0]]).transpose()
    assert (model.model(dt=dt, state=state_vector).transpose() == [0.0, 1.1]).all()

    state_vector = np.array([[1.0, 0.0]]).transpose()
    assert (model.model(dt=dt, state=state_vector).transpose() == [0.0, 0.1]).all()

    state_vector = np.array([[1.0, 1.0]]).transpose()
    assert (model.model(dt=dt, state=state_vector).transpose() == [1.0, 1.1]).all()


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
            dt=dt, state=state_vector, control_vector=control_vector
        ).transpose(),
        [[0.0, 0.02]],
    )

    state_vector = np.array([[0.0, 1.0]]).transpose()
    assert_almost_equal(
        model.model(
            dt=dt, state=state_vector, control_vector=control_vector
        ).transpose(),
        [[0.0, 1.02]],
    )

    state_vector = np.array([[1.0, 0.0]]).transpose()
    assert_almost_equal(
        model.model(
            dt=dt, state=state_vector, control_vector=control_vector
        ).transpose(),
        [[0.0, 0.02]],
    )

    state_vector = np.array([[1.0, 1.0]]).transpose()
    assert_almost_equal(
        model.model(
            dt=dt, state=state_vector, control_vector=control_vector
        ).transpose(),
        [[1.0, 1.02]],
    )


@given(floats(), floats(), floats())
def test_Model_impl_property(x, y, a):
    config = {}
    dt = 0.1

    ui_Model = ui.Model(
        ui.Symbol("dt"),
        set(ui.symbols(["x", "y"])),
        set(ui.symbols(["a"])),
        {ui.Symbol("x"): ui.Symbol("x") * ui.Symbol("y"), ui.Symbol("y"): "y + a * dt"},
    )
    model = python.Model(
        ui_Model,
        config,
    )

    control_vector = np.array([[a]])
    state_vector = np.array([[x, y]]).transpose()
    if not np.isfinite(state_vector).all() or not np.isfinite(control_vector).all():
        reject()
    if (np.abs(state_vector) > 1e100).any() or (np.abs(control_vector) > 1e100).any():
        reject()

    next_state = model.model(dt=dt, state=state_vector, control_vector=control_vector)

    for i, key in enumerate(model.arglist[1 : model.state_size + 1]):
        python_version = next_state[i]

        subs_args = [
            (symbol, float(val))
            for (symbol, val) in zip(
                model.arglist, [dt] + list(state_vector) + list(control_vector)
            )
        ]
        print(subs_args)
        symbolic_version = ui_Model.state_model[key].subs(subs_args)

        assert_almost_equal(python_version, symbolic_version)
