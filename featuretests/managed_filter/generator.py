from formak import cpp
from formak.problemdefinition.model import Model as UiModel
from sympy import Symbol

dt = Symbol("dt")

tp = trajectory_properties = {k: Symbol(k) for k in ["mass", "z", "v", "a"]}

thrust = Symbol("thrust")

state = set(tp.values())
control = {thrust}

state_model = {
    tp["mass"]: tp["mass"],
    tp["z"]: tp["z"] + dt * tp["v"],
    tp["v"]: tp["v"] + dt * tp["a"],
    tp["a"]: -9.81 * tp["mass"] + thrust,
}

model = UiModel(dt=dt, state=state, control=control, state_model=state_model)

cpp_implementation = cpp.compile_ekf(
    state_model=model,
    process_noise={thrust: 1.0},
    sensor_models={
        "simple": {Symbol("v"): Symbol("v")},
        "accel": {Symbol("a"): Symbol("a")},
    },
    sensor_noises={"simple": {tp["v"]: 1.0}, "accel": {tp["a"]: 1.0}},
    config={"common_subexpression_elimination": True, "max_dt_sec": 0.05},
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
