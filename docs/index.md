# FormaK

FormaK builds fast, reliable, easy to use state estimation.

FormaK is tooling to easily derive Python and C++ implementations of models
based on your data (or even before you have data). The models are fast and are
generated with automation to help you generate and maintain them. FormaK uses
symbolic mathematics for fast, efficient system modelling and applies compiler
techniques to code generation to create performant code that is easy to use.

The values of the FormaK project (in order) are:

- Easy to use
- Performant

FormaK is open source and uses the MIT license.

If you'd like to jump in to using FormaK: [Getting Started](getting-started.html)

The code is hosted on Github: [github.com/buckbaskin/formak](https://github.com/buckbaskin/formak)

### The Persona

Who is this for?

The user of FormaK is someone with domain expertise who is looking to take
their knowledge and their data and quickly (measured in the user's time) create
a model, either as a project in itself or as part of a larger project.

The user isn't expected to know or have to understand the mechanics of what's
going on under the hood in order to get value from the library. This means
things like sane defaults and an easy to use interface that's hard to misuse
are highly important. The library should encapsulate a collective knowledge
that, as it improves over time, can improve everyone's work.

On the flip side, a user with more familiarity of the modelling process or the
FormaK tool should be able to use more advanced features and select
configuration that better matches their use case.

### The Five Keys

In line with the values and the intended user, the intended user experience is
as follows. The user provides:

- Model that describes the physics of the system
- Execution criteria (e.g. memory usage, execution time)
- Time series data for the system

and in return the user gets an optimal model.

The Five Key Elements the library provides to achieve this user experience are:

1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

## Example

What does this look like?

```python

vp = vehicle_properties = {k: Symbol(k) for k in ["m", "x", "v", "a"]}
fuel_burn_rate = Symbol("fuel_burn_rate")

state = set(vehicle_properties.values())

control = set([fuel_burn_rate])  # kg/sec

# momentum = mv
# dmomentum / dt = F = d(mv)/dt
# F = m dv/dt + dm/dt v
# a = dv / dt = (F - dm/dt * v) / m

F = -gravitational_force(vp["m"], Earth_Mass, vp["x"] + Earth_Equatorial_Radius)

state_model = {
    vp["m"]: vp["m"] - fuel_burn_rate * dt,
    vp["x"]: vp["x"] + (vp["v"] * dt) + (1 / 2 * vp["a"] * dt * dt),
    vp["v"]: vp["v"] + (vp["a"] * dt),
    vp["a"]: (F - (fuel_burn_rate * vp["v"])) / vp["m"],
}

orbital_model = Model(dt=dt, state=state, control=control, state_model=state_model)

python_implementation = python.compile(orbital_model)
```

## Features

- Tools for creating models
- Optimize the models
- Generate Python from models
- Generate C++ from models
- (planned) Integrations for model fitting, model selection

## Requirements

- Bazel
- Clang-12 / C++17
- Python3
- pip
