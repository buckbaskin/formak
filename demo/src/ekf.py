"""
# EKF

This file demonstrates defining a symbolic model with some simple approximate
physics. Then, compile it into a Python EKF implementation

The same symbolic model could also be compiled into a C++ implementation.
"""

from formak.ui import Model, symbols, Symbol
from formak import python

from collections import defaultdict

dt = symbols("dt")  # change in time


def main():
    """
    The main function encapsulates a few common steps for all models:
    1. Defining symbols
    2. Identifying the symbol(s) used in the model state
    3. Identifying the symbol(s) used in the control inputs
    4. Defining the discrete state update
    5. Constructing the Model class

    The file also demonstrates:
    6. Compiling from the symbolic class to a Python model implementation
    7. Running a model update with the Python model
    """

    # 1. Defining symbols
    vp = vehicle_properties = {k: Symbol(k) for k in ["m", "x", "v", "a"]}
    fuel_burn_rate = Symbol("fuel_burn_rate")

    # 2. Identifying the symbol(s) used in the model state
    state = set(vehicle_properties.values())

    # 3. Identifying the symbol(s) used in the control inputs
    control = {fuel_burn_rate}  # kg/sec

    # 4. Defining the discrete state update

    # momentum = mv
    # dmomentum / dt = F = d(mv)/dt
    # F = m dv/dt + dm/dt v
    # a = dv / dt = (F - dm/dt * v) / m

    F = 9.81

    state_model = {
        vp["m"]: vp["m"] - fuel_burn_rate * dt,
        vp["x"]: vp["x"] + (vp["v"] * dt) + (1 / 2 * vp["a"] * dt * dt),
        vp["v"]: vp["v"] + (vp["a"] * dt),
        vp["a"]: (F - (fuel_burn_rate * vp["v"])) / vp["m"],
    }

    # 5. Constructing the Model class
    symbolic_model = Model(dt, state, control, state_model, debug_print=True)

    initial_state = {
        vp["m"]: 10.0,
        vp["x"]: 0.0,
        vp["v"]: 0.0,
        vp["a"]: 0.0,
    }

    # 6. Compiling from the symbolic class to a Python EKF implementation
    python_ekf = python.compile_ekf(
        symbolic_model=symbolic_model,
        process_noise={fuel_burn_rate: 1.0},
        sensor_models={"simple": {vp["v"]: vp["v"]}},
        sensor_noises={"simple": {vp["v"]: 1.0}},
    )

    # 7. Running a process model update with the Python EKF
    state_vector = python_ekf.State.from_dict(initial_state)
    control_vector = python_ekf.Control.from_dict({fuel_burn_rate: 0.01})
    state_variance = python_ekf.Covariance()

    print("Initial State")
    print(state_vector)

    state_vector_next, state_variance_next = python_ekf.process_model(
        0.1, state_vector, state_variance, control_vector
    )

    print("Next State")
    print(state_vector_next)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
