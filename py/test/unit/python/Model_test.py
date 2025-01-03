import warnings

from numpy.testing import assert_almost_equal

from formak import python
from formak.problemdefinition.model import Model as UiModel
from sympy import Symbol, symbols

warnings.filterwarnings("error")


def test_Model_creation_list():
    dt = Symbol("dt")
    model = python.Model(UiModel(dt, [], [], {}), {})

    assert model.arglist == symbols(["dt"])


def test_Model_creation_set():
    dt = Symbol("dt")
    model = python.Model(UiModel(dt, set(), set(), {}), {})

    assert model.arglist == symbols(["dt"])

    model = python.Model(
        UiModel(dt, set(symbols(["x"])), set(), {Symbol("x"): "x"}), {}
    )

    assert model.arglist == symbols(["dt", "x"])

    model = python.Model(
        UiModel(
            dt,
            set(symbols(["x", "y"])),
            set(),
            {Symbol("x"): "x", Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == symbols(["dt", "x", "y"])

    model = python.Model(UiModel(dt, set(), set(symbols(["a"])), {}), {})

    assert model.arglist == symbols(["dt", "a"])

    model = python.Model(
        UiModel(dt, set(symbols(["x"])), set(symbols(["a"])), {Symbol("x"): "x"}),
        {},
    )

    assert model.arglist == symbols(["dt", "x", "a"])

    model = python.Model(
        UiModel(
            dt,
            set(symbols(["x", "y"])),
            set(symbols(["a"])),
            {Symbol("x"): "x", Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == symbols(["dt", "x", "y", "a"])

    model = python.Model(UiModel(dt, set(), set(symbols(["a", "b"])), {}), {})

    assert model.arglist == symbols(["dt", "a", "b"])

    model = python.Model(
        UiModel(
            dt,
            set(symbols(["x"])),
            set(symbols(["a", "b"])),
            {Symbol("x"): "x"},
        ),
        {},
    )

    assert model.arglist == symbols(["dt", "x", "a", "b"])

    model = python.Model(
        UiModel(
            dt,
            set(symbols(["x", "y"])),
            set(symbols(["a", "b"])),
            {Symbol("x"): "x", Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == symbols(["dt", "x", "y", "a", "b"])


def test_Model_impl_no_control():
    config = {}
    dt = 0.1

    model = python.Model(
        UiModel(
            Symbol("dt"),
            set(symbols(["x", "y"])),
            set(),
            {Symbol("x"): "x * y", Symbol("y"): "y + 0.1"},
        ),
        config,
    )

    state_vector = model.State()
    assert (model.model(dt=dt, state=state_vector).data.transpose() == [0.0, 0.1]).all()

    state_vector = model.State(y=1.0)
    assert (model.model(dt=dt, state=state_vector).data.transpose() == [0.0, 1.1]).all()

    state_vector = model.State(x=1.0)
    assert (model.model(dt=dt, state=state_vector).data.transpose() == [0.0, 0.1]).all()

    state_vector = model.State(x=1.0, y=1.0)
    assert (model.model(dt=dt, state=state_vector).data.transpose() == [1.0, 1.1]).all()


def test_Model_impl_control():
    config = {}
    dt = 0.1

    model = python.Model(
        UiModel(
            Symbol("dt"),
            set(symbols(["x", "y"])),
            set(symbols(["a"])),
            {Symbol("x"): "x * y", Symbol("y"): "y + a * dt"},
        ),
        config,
    )

    control = model.Control(a=0.2)

    state_vector = model.State()
    assert_almost_equal(
        model.model(dt=dt, state=state_vector, control=control).data.transpose(),
        [[0.0, 0.02]],
    )

    state_vector = model.State(y=1.0)
    assert_almost_equal(
        model.model(dt=dt, state=state_vector, control=control).data.transpose(),
        [[0.0, 1.02]],
    )

    state_vector = model.State(x=1.0)
    assert_almost_equal(
        model.model(dt=dt, state=state_vector, control=control).data.transpose(),
        [[0.0, 0.02]],
    )

    state_vector = model.State(x=1.0, y=1.0)
    assert_almost_equal(
        model.model(dt=dt, state=state_vector, control=control).data.transpose(),
        [[1.0, 1.02]],
    )
