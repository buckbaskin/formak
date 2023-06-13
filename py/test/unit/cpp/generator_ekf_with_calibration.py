import numpy as np

from formak import cpp, ui

dt, a, b, x, y = ui.symbols(["dt", "a", "b", "x", "y"])

ui_model = ui.Model(
    dt=dt,
    state=set([x]),
    control=set(),
    calibration=set([a, b]),
    state_model={x: x + a + b},
)

cpp_implementation = cpp.compile_ekf(
    state_model=ui_model,
    process_noise={},
    sensor_models={"y": {y: x + b}},
    sensor_noises={"y": np.eye(1)},
    calibration_map={ui.Symbol("a"): 5.0, ui.Symbol("b"): 0.5},
    config={},
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
