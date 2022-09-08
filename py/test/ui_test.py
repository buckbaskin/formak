from formak.ui import *


def test_model_simplification():
    x = Symbol("x")
    model = Model(set([x]), set(), {x: x * x / x})

    assert model.state_model[x] == x


if __name__ == "__main__":
    import sys
    import pytest as test_runner

    # sys.exit(test_runner.main(sys.argv[1:]))
    sys.exit(test_model_simplification())
