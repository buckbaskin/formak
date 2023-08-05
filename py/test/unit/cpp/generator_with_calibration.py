from formak import cpp, ui

dt, a, b, x = ui.symbols(["dt", "a", "b", "x"])

ui_model = ui.Model(
    dt=dt,
    state={x},
    control=set(),
    calibration={a, b},
    state_model={x: x + a + b},
)

cpp_implementation = cpp.compile(
    ui_model, calibration_map={ui.Symbol("a"): 5.0, ui.Symbol("b"): 0.5}, config={}
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
