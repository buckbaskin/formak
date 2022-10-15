from formak.ui import Symbol, Model


def test_model_simplification():
    dt = Symbol("dt")
    x = Symbol("x")
    model = Model(dt, set([x]), set(), {x: x * x / x})

    assert model.state_model[x] == x
