import sys
from os.path import basename

from formak.ui import *
from formak import cpp

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

model = Model(dt=dt, state=state, control=control, state_model=state_model)

cpp_implementation = cpp.compile(model)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
