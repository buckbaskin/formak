from typing import Dict
from formak import ui

dt = ui.Symbol("dt")

tp = _trajectory_properties = {k: ui.Symbol(k) for k in ["mass", "z", "v", "a"]}

thrust = ui.Symbol("thrust")

state = set(tp.values())
control = {thrust}

state_model = {
    tp["mass"]: tp["mass"],
    tp["z"]: tp["z"] + dt * tp["v"],
    tp["v"]: tp["v"] + dt * tp["a"],
    tp["a"]: -9.81 * tp["mass"] + thrust,
}

symbolic_model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

process_noise = {thrust: 0.01}
sensor_models = {"velocity": {tp["v"]: tp["v"]}}
sensor_noises = {"velocity": {tp["v"]: 1.0}}
# TODO add a calibration term here (mass?)
calibration_map = {} # type: Dict[str, float]
