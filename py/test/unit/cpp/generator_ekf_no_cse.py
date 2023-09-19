from formak import cpp, ui

cpp_implementation = cpp.compile_ekf(
    state_model=ui.Model(
        dt=ui.Symbol("dt"),
        state=set(ui.symbols(["x", "y"])),
        control=set(ui.symbols(["a"])),
        # Add 1e-3 * a to prevent (0, 0) from having no variance
        state_model={
            ui.Symbol("x"): "x + y * a * dt",
            ui.Symbol("y"): "y + x * a * dt",
        },
    ),
    process_noise={ui.Symbol("a"): 0.25},
    sensor_models={
        "simple": {"reading1": ui.Symbol("x")},
        "combined": {"reading2": ui.Symbol("x") + ui.Symbol("y")},
    },
    sensor_noises={"simple": {"reading1": 1.0}, "combined": {"reading2": 4.0}},
    config=cpp.Config(common_subexpression_elimination=False),
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
