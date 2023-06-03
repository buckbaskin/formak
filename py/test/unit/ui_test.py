from formak.ui import Model, Symbol


def test_model_simplification():
    dt = Symbol("dt")
    x = Symbol("x")
    model = Model(dt, {x}, set(), {x: x * x / x})

    assert model.state_model[x] == x
