from formak import ui, python


def test_Model_creation_list():
    model = python.Model(ui.Model([], [], {}), {})

    assert model.arglist == ui.symbols([])


def test_Model_creation_set():
    model = python.Model(ui.Model(set(), set(), {}), {})

    assert model.arglist == ui.symbols([])

    model = python.Model(
        ui.Model(set(ui.symbols(["x"])), set(), {ui.Symbol("x"): "x"}), {}
    )

    assert model.arglist == ui.symbols(["x"])

    model = python.Model(
        ui.Model(
            set(ui.symbols(["x", "y"])),
            set(),
            {ui.Symbol("x"): "x", ui.Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == ui.symbols(["x", "y"])

    model = python.Model(ui.Model(set(), set(ui.symbols(["a"])), {}), {})

    assert model.arglist == ui.symbols(["a"])

    model = python.Model(
        ui.Model(set(ui.symbols(["x"])), set(ui.symbols(["a"])), {ui.Symbol("x"): "x"}),
        {},
    )

    assert model.arglist == ui.symbols(["x", "a"])

    model = python.Model(
        ui.Model(
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x", ui.Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == ui.symbols(["x", "y", "a"])

    model = python.Model(ui.Model(set(), set(ui.symbols(["a", "b"])), {}), {})

    assert model.arglist == ui.symbols(["a", "b"])

    model = python.Model(
        ui.Model(
            set(ui.symbols(["x"])), set(ui.symbols(["a", "b"])), {ui.Symbol("x"): "x"}
        ),
        {},
    )

    assert model.arglist == ui.symbols(["x", "a", "b"])

    model = python.Model(
        ui.Model(
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a", "b"])),
            {ui.Symbol("x"): "x", ui.Symbol("y"): "y"},
        ),
        {},
    )

    assert model.arglist == ui.symbols(["x", "y", "a", "b"])


if __name__ == "__main__":
    import sys
    import pytest as test_runner

    sys.exit(test_runner.main(sys.argv[1:]))
