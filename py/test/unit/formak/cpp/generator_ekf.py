import numpy as np

from formak import cpp, ui

cpp_implementation = cpp.compile_ekf(
    state_model=ui.Model(
        dt=ui.Symbol("dt"),
        state=set(ui.symbols(["x", "y"])),
        control=set(ui.symbols(["a"])),
        # Add 1e-3 * a to prevent (0, 0) from having no variance
        state_model={ui.Symbol("x"): "x + y * dt", ui.Symbol("y"): "y + a * dt"},
    ),
    process_noise=np.eye(1) * 0.25,
    sensor_models={
        "simple": {"reading1": ui.Symbol("x")},
        "combined": {"reading2": ui.Symbol("x") + ui.Symbol("y")},
    },
    sensor_noises={"simple": np.eye(1), "combined": np.eye(1) * 4.0},
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
