"""
Innovation Filtering Featuretest

Create a model with states:
- x, y, heading, velocity model

Provide heading readings, expect rejecting 180 degree heading errors.
Nonlinear model provides clear divergence signal. If innovation filtering isn't
working as expected, then the model will flip into the wrong direction.
"""
from math import degrees, radians

from sympy import cos, sin

from formak import cpp, ui

TRUE_SCALE = radians(5.0)

dt = ui.Symbol("dt")

x, y, heading, velocity, _heading_err = ui.symbols(
    ["x", "y", "heading", "velocity", "_heading_err"]
)
state = {x, y, heading}
control = {velocity, _heading_err}

state_model = {
    x: x + dt * velocity * cos(heading),
    y: y + dt * velocity * sin(heading),
    heading: heading + _heading_err,
}

model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

config = cpp.Config(innovation_filtering=4)

cpp_implementation = cpp.compile_ekf(
    state_model=model,
    process_noise={velocity: 1.0, _heading_err: 0.1},
    sensor_models={"compass": {heading: heading}},
    sensor_noises={"compass": {heading: TRUE_SCALE}},
    config=config,
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
