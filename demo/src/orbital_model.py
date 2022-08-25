import sympy

from collections import defaultdict
from sympy import symbols, Symbol, diff, simplify

dt = symbols("dt")  # change in time

G = gravitational_constant = 6.674e-11  # m^3 / kg / s**2
Earth_Mass = 5.9722e24  # kg
Earth_Equatorial_Radius = 6378.137  # m


def SaturnVMass():
    # Apollo 7
    masses = {
        "SIB_dry": 84530,
        "SIB_oxidizer": 631300,
        "SIB_fuel": 276900,
        "SIB_other": 1182,
        "SIBSIVBinterstage_dry": 5543,
        "SIBSIVBinterstage_propellant": 1061,
        "SIVB_dry": 21852,
        'SIVB_oxidizer': 193330,
        'SIVB_fuel': 39909,
        'SIVB_other': 1432,
        'Instrument_dry': 4263,
        'Spacecraft_dry': 45312,
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
        group = mass_description.split('_')[0]
        mass_groups[group] += mass_kg

    print('Mass Groups')
    for k, v in sorted(list(mass_groups.items())):
        print('  %s: %s' % (k, v))

    return {"dry": dry, "consumable": consumable}


Vehicle_Mass_Properties = SaturnVMass()


# states, calibrates, constants


def gravitational_force(m_1, m_2, r):
    return G * (m_1 * m_2) / (r ** 2)


def main():
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

    for k in list(state_model.keys()):
        state_model[k] = simplify(state_model[k])

    assert len(state_model) == len(state)
    for k in state:
        try:
            assert k in state_model
        except AssertionError:
            print("%s ( %s ) missing from state model" % (k, type(k)))
            raise

    initial_state = {
        vp["m"]: Vehicle_Mass_Properties["dry"] + Vehicle_Mass_Properties["consumable"],
        vp["x"]: 0.0,
        vp["v"]: 0.0,
        vp["a"]: 0.0,
    }

    assert len(state_model) == len(state)
    for k in state:
        try:
            assert k in state_model
        except AssertionError:
            print("%s ( %s ) missing from state model" % (k, type(k)))
            raise

    print("State Model")
    for k in sorted(list(state_model.keys()), key=lambda x: x.name):
        print("  %s: %s" % (k, state_model[k]))

    print("Initial State")
    for k in sorted(list(initial_state.keys()), key=lambda x: x.name):
        print("  %s: %s" % (k, initial_state[k]))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
