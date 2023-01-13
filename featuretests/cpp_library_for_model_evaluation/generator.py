import sys
from os.path import basename

print("sys.path matching formak as dep")
print(list(filter(lambda p: "formak" in basename(p).lower(), list(sys.path))))

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

# TODO(buck): Some of this should be split out into FormaK libraries instead of the user script (much of this? all of this? probably everything but the ui.Model creation)
# TODO(buck): It should end up looking like:
cpp_implementation = cpp.compile(model)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
