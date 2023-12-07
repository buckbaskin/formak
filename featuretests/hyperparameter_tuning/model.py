from formak import ui

dt = ui.Symbol("dt")

tp = _trajectory_properties = {k: ui.Symbol(k) for k in ["z", "v", "a"]}

thrust = ui.Symbol("thrust")
mass = ui.Symbol("mass")

state = set(tp.values())
control = {thrust}

state_model = {
    tp["z"]: tp["z"] + dt * tp["v"],
    tp["v"]: tp["v"] + dt * tp["a"],
    tp["a"]: -9.81 * mass + thrust,
}

symbolic_model = ui.Model(
    dt=dt,
    state=state,
    control=control,
    state_model=state_model,
    calibration=set([mass]),
)

process_noise = {thrust: 0.01}
sensor_models = {"velocity": {tp["v"]: tp["v"]}}
sensor_noises = {"velocity": {tp["v"]: 1.0}}
calibration_map = {mass: 0.0}
