"""
Feature Test.

Run the new ManagedFilter with a generated EKF. Run it with and without sensor readings.

Passes if the ManagedFilter updates the state (other testing checks if it's
working correctly).
"""

import numpy as np
from formak.runtime import ManagedFilter, StampedReading

from formak import python, ui


def make_ekf():
    dt = ui.Symbol("dt")

    tp = {k: ui.Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = ui.Symbol("thrust")

    state = set(tp.values())
    control = {thrust}

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    v = ui.Symbol("v")

    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

    ekf = python.compile_ekf(
        symbolic_model=model,
        process_noise={thrust: 1.0},
        sensor_models={"simple": {v: v}},
        sensor_noises={"simple": {v: 1.0}},
    )
    return ekf


def test_tick_time_only():
    ekf = make_ekf()
    state = ekf.State(v=1.0)
    covariance = ekf.Covariance()
    control = ekf.Control()
    mf = ManagedFilter(ekf=ekf, start_time=0.0, state=state, covariance=covariance)

    state0p1 = mf.tick(0.1, control=control)

    state0p2 = mf.tick(0.2, control=control)

    assert np.any(state0p1.state != state0p2.state)


def test_tick_empty_sensor_readings():
    ekf = make_ekf()
    state = ekf.State(v=1.0)
    covariance = ekf.Covariance()
    control = ekf.Control()
    mf = ManagedFilter(ekf=ekf, start_time=2.0, state=state, covariance=covariance)

    state0p1 = mf.tick(2.1, control=control, readings=[])

    state0p2 = mf.tick(2.2, control=control, readings=[])

    assert np.any(state0p1.state != state0p2.state)


def test_tick_one_sensor_reading():
    ekf = make_ekf()
    state = ekf.State(v=1.0)
    covariance = ekf.Covariance()
    control = ekf.Control()
    mf = ManagedFilter(ekf=ekf, start_time=2.0, state=state, covariance=covariance)

    reading1 = StampedReading(2.05, "simple")

    state0p1 = mf.tick(2.1, control=control, readings=[reading1])

    reading2 = StampedReading(2.15, "simple")

    state0p2 = mf.tick(2.2, control=control, readings=[reading2])

    assert np.any(state0p1.state != state0p2.state)


def test_tick_multiple_sensor_reading():
    ekf = make_ekf()
    state = ekf.State(v=1.0)
    covariance = ekf.Covariance()
    control = ekf.Control()
    mf = ManagedFilter(ekf=ekf, start_time=3.0, state=state, covariance=covariance)

    readings1 = [
        StampedReading(3.05, "simple"),
        StampedReading(3.06, "simple"),
        StampedReading(3.07, "simple"),
    ]

    state0p1 = mf.tick(3.1, control=control, readings=readings1)

    readings2 = [
        StampedReading(3.15, "simple"),
        StampedReading(3.16, "simple"),
        StampedReading(3.17, "simple"),
    ]

    state0p2 = mf.tick(3.2, control=control, readings=readings2)

    assert np.any(state0p1.state != state0p2.state)
