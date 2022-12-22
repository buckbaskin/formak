# Getting Started for Users

Check out `demo/src/orbital_model.py` for an example model and use.

Some highlights:

```py
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

    model = Model(dt, state, control, state_model, debug_print=True)
```

More coming soon!

# Getting Started For Developers

## Installation

This project uses Bazel as its build system. To get started, make sure you have
Bazelisk, Python3 and Clang available.

### Requirements

- Bazel
- Clang-12 / C++17
- Python3


### Set up Bazelisk

### Install Clang

## Running Some Code

To get started running code for the project, try the command

`make ci`

This will run all of the unit tests for the project and if it passes it indicates that the project is set up correctly

### Common Issues

...

### Next Steps

Using bazel you can specify a more fine-grained set of code to run. For example, if you're interested in the compilation feature available in Python generation, you can run the command

`bazel test //featuretests:python-library-for-model-evaluation`
