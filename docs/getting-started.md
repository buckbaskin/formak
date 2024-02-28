# Getting Started for Users

Check out `demo/src/symbolic_model.py` for an example model and use.

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

The above code can be extended with the `DesignManager` to help select
appropriate parameters.

```py
    manager = DesignManager(name="mercury")

    # Define the symbolic model
    manager = manager.symbolic_model(model=model)

    # Select model parameters, such as innovation_filtering from data
    manager = symbolic_model_state.fit_model(
        parameter_space={
            "process_noise": [process_noise],
            "sensor_models": [sensor_models],
            "sensor_noises": [sensor_noises],
            "calibration_map": [calibration_map],
            "innovation_filtering": [None, 1, 2, 3, 4, 5, 6, 7],
        },
        data=data,
    )

    # Export the refined model. Note: not a state transition
    python_model = manager.export_python()
```

Check out `demo/` for additional examples and `featuretests/` for examples of
specific features.

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

### Install Python Developer Tooling

`python3 -m pip install -U -r requirements_dev.txt`

This will install the dependencies necessary for helper scripts like:
- `make format`
- `make lint`

## Running Some Code

To get started running code for the project, try the command

`make ci`

This will run all of the unit tests for the project and if it passes it indicates that the project is set up correctly

### Common Issues

...

### Next Steps

Using bazel you can specify a more fine-grained set of code to run. For example, if you're interested in the compilation feature available in Python generation, you can run the command

`bazel test //featuretests:python-library-for-model-evaluation`
