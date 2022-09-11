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


if __name__ == "__main__":
    import sys
    import pytest as test_runner

    sys.exit(test_runner.main(sys.argv[1:]))
