from formak.ui import *


def test_model_simplification():
    x = Symbol("x")
    model = Model(set([x]), set(), {x: x * x / x})

    assert model.state_model[x] == x
