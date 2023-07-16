import numpy as np
import pytest

from formak import runtime


def test_tick_time_only():
    mf = runtime.ManagedFilter()

    state0p1 = mf.tick(0.1)

    state0p2 = mf.tick(0.2)

    assert state0p1 != state0p2


def test_tick_empty_sensor_readings():
    mf = runtime.ManagedFilter()

    state0p1 = mf.tick(0.1, [])

    state0p2 = mf.tick(0.2, [])

    assert state0p1 != state0p2


def test_tick_one_sensor_reading():
    mf = runtime.ManagedFilter()

    reading1 = None

    state0p1 = mf.tick(0.1, [reading1])

    reading2 = None

    state0p2 = mf.tick(0.2, [reading2])

    assert state0p1 != state0p2


def test_tick_multiple_sensor_reading():
    mf = runtime.ManagedFilter()

    readings1 = [None, None, None]

    state0p1 = mf.tick(0.1, readings1)

    readings2 = [None, None, None]

    state0p2 = mf.tick(0.2, readings2)

    assert state0p1 != state0p2
