"""Define elements of a common model with simple dynamics to demonstrate parameter fitting."""

from formak.ui.model import Model as UiModel
from sympy import Symbol

dt = Symbol("dt")

tp = _trajectory_properties = {k: Symbol(k) for k in ["z", "v", "a"]}

thrust = Symbol("thrust")
mass = Symbol("mass")

state = set(tp.values())
control = {thrust}

state_model = {
    tp["z"]: tp["z"] + dt * tp["v"],
    tp["v"]: tp["v"] + dt * tp["a"],
    tp["a"]: -9.81 * mass + thrust,
}

symbolic_model = UiModel(
    dt=dt,
    state=state,
    control=control,
    state_model=state_model,
    calibration={mass},
)

process_noise = {thrust: 0.01}
sensor_models = {"velocity": {tp["v"]: tp["v"]}}
sensor_noises = {"velocity": {tp["v"]: 1.0}}
calibration_map = {mass: 0.0}
