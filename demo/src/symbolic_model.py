"""
# Symbolic Model

This file demonstrates defining a symbolic model with some simple approximate
physics. To use this model, it is compiled into a Python implementation.

The same symbolic model could also be compiled into a C++ implementation.

See featuretests/python_library_for_model_evaluation/simple_test.py and the
featuretests/ directory for additional features and examples.
"""

from formak.ui import Model, symbols, Symbol
from formak import python

from collections import defaultdict

dt = symbols("dt")  # change in time

G = gravitational_constant = 6.674e-11  # m^3 / kg / s**2
Earth_Mass = 5.9722e24  # kg
Earth_Equatorial_Radius = 6378.137  # m


def SaturnVMass():
    """
    Summarize the mass distribution from the Apollo 7 launch vehicle. This mass
    will be used to initialize the state of the model.

    The details of the exact masses don't impact the definition of the model
    """
    # Apollo 7
    masses = {
        "SIB_dry": 84530,
        "SIB_oxidizer": 631300,
        "SIB_fuel": 276900,
        "SIB_other": 1182,
        "SIBSIVBinterstage_dry": 5543,
        "SIBSIVBinterstage_propellant": 1061,
        "SIVB_dry": 21852,
        "SIVB_oxidizer": 193330,
        "SIVB_fuel": 39909,
        "SIVB_other": 1432,
        "Instrument_dry": 4263,
        "Spacecraft_dry": 45312,
    }

    dry = 0
    consumable = 0

    for mass_description, mass_kg in masses.items():
        if "dry" in mass_description:
            dry += mass_kg
        else:
            consumable += mass_kg

    assert (dry + consumable) == sum(masses.values())

    mass_groups = defaultdict(float)
    for mass_description, mass_kg in masses.items():
        group = mass_description.split("_")[0]
        mass_groups[group] += mass_kg

    print("Mass Groups")
    for k, v in sorted(list(mass_groups.items())):
        print("  {}: {}".format(k, v))

    return {"dry": dry, "consumable": consumable}


Vehicle_Mass_Properties = SaturnVMass()


# states, calibrates, constants


def gravitational_force(m_1, m_2, r):
    """
    Calculate the gravitational force acting between two bodies.

    This function can be called with floating point values or symbolic values
    representing the possible masses and radii between the two masses. For this
    example, it's used to encapsulate the calculation of the gravitational
    force acting on the simplified model of a launch vehicle.
    """
    return -G * (m_1 * m_2) / (r**2)


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

    F = gravitational_force(vp["m"], Earth_Mass, vp["x"] + Earth_Equatorial_Radius)

    state_model = {
        vp["m"]: vp["m"] - fuel_burn_rate * dt,
        vp["x"]: vp["x"] + (vp["v"] * dt) + (1 / 2 * vp["a"] * dt * dt),
        vp["v"]: vp["v"] + (vp["a"] * dt),
        vp["a"]: (F - (fuel_burn_rate * vp["v"])) / vp["m"],
    }

    # 5. Constructing the Model class
    symbolic_model = Model(dt, state, control, state_model, debug_print=True)

    initial_state = {
        vp["m"]: Vehicle_Mass_Properties["dry"] + Vehicle_Mass_Properties["consumable"],
        vp["x"]: 0.0,
        vp["v"]: 0.0,
        vp["a"]: 0.0,
    }

    # 6. Compiling from the symbolic class to a Python model implementation
    python_model = python.compile(symbolic_model=symbolic_model)

    # 7. Running a model update with the Python model
    state_vector = python_model.State.from_dict(initial_state)
    control_vector = python_model.Control.from_dict({fuel_burn_rate: 0.0})

    print("Initial State")
    print(state_vector)

    state_vector_next = python_model.model(
        dt=0.1, state=state_vector, control=control_vector
    )

    print("Next State")
    print(state_vector_next)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
