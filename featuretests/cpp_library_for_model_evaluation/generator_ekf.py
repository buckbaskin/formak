from formak import cpp, ui

dt = ui.Symbol("dt")

tp = trajectory_properties = {k: ui.Symbol(k) for k in ["mass", "z", "v", "a"]}

thrust = ui.Symbol("thrust")

state = set(tp.values())
control = {thrust}

state_model = {
    tp["mass"]: tp["mass"],
    tp["z"]: tp["z"] + dt * tp["v"],
    tp["v"]: tp["v"] + dt * tp["a"],
    tp["a"]: -9.81 * tp["mass"] + thrust,
}

v = ui.Symbol("v")

model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

cpp_implementation = cpp.compile_ekf(
    state_model=model,
    process_noise={thrust: 1.0},
    sensor_models={"simple": {v: v}},
    sensor_noises={"simple": {v: 1.0}},
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
