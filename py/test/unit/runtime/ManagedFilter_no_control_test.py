from enum import Enum, auto
from typing import List, Tuple

import numpy as np
from formak.runtime import ManagedFilter, StampedReading
from hypothesis import given, settings
from hypothesis.strategies import permutations, sampled_from

from formak import python, ui


def samples_dt_sec():
    return [0.0, 0.1, -0.1, -1.5, 2.7]


class ShuffleId(Enum):
    Zero = 0
    Sensor0 = auto()
    Sensor1 = auto()
    Sensor2 = auto()
    Sensor3 = auto()
    Output = auto()


def parse_options(
    output_dt: float, shuffle: List[ShuffleId]
) -> Tuple[float, List[float]]:
    options = [0.0, 0.0, 0.0, 0.0]
    for i, elem in enumerate(shuffle):
        index_dt = i * 0.1
        if elem == ShuffleId.Zero:
            output_dt -= index_dt
            for j in range(4):
                options[j] -= index_dt
        elif elem == ShuffleId.Output:
            output_dt += index_dt
        elif elem == ShuffleId.Sensor0:
            options[0] += index_dt
        elif elem == ShuffleId.Sensor1:
            options[1] += index_dt
        elif elem == ShuffleId.Sensor2:
            options[2] += index_dt
        elif elem == ShuffleId.Sensor3:
            options[3] += index_dt
        else:
            raise ValueError(f"parse_options got invalid element: {elem} at index {i}")
    return output_dt, options


def make_ekf(calibration_map):
    dt = ui.Symbol("dt")

    state = ui.Symbol("state")

    calibration_velocity = ui.Symbol("calibration_velocity")

    state_model = {state: state + dt * (calibration_velocity)}

    state_set = {state}

    model = ui.Model(
        dt=dt,
        state=state_set,
        control=set(),
        state_model=state_model,
        calibration={calibration_velocity},
    )

    ekf = python.compile_ekf(
        symbolic_model=model,
        process_noise={},
        sensor_models={"simple": {state: state}},
        sensor_noises={"simple": {state: 1e-9}},
        calibration_map=calibration_map,
        config={"innovation_filtering": None},
    )
    return ekf


def test_constructor():
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map)
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)
    _mf = ManagedFilter(ekf=ekf, start_time=0.0, state=state, covariance=covariance)


@settings(deadline=None)
@given(sampled_from(samples_dt_sec()))
def test_tick_no_readings(dt):
    start_time = 10.0
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map=calibration_map)
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    state0p1 = mf.tick(start_time + dt)

    print("state")
    print(state0p1.state)
    print("reading")
    print(state, dt)
    print("diff")
    print((state0p1.state.data) - (state.data))
    assert np.isclose(state0p1.state.data, state.data, atol=2.0e-14).all()
    assert state0p1.covariance.data == covariance.data


@settings(deadline=None)
@given(sampled_from(samples_dt_sec()))
def test_tick_empty_readings(dt):
    start_time = 10.0
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map=calibration_map)
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    state0p1 = mf.tick(start_time + dt, readings=[])

    assert np.isclose(state0p1.state.data, state.data, atol=2.0e-14).all()
    assert state0p1.covariance.data == covariance.data


@settings(deadline=None)
@given(sampled_from(samples_dt_sec()), sampled_from(samples_dt_sec()))
def test_tick_one_reading(output_dt, reading_dt):
    start_time = 10.0
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map=calibration_map)
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    reading_v = -3.0
    reading1 = StampedReading(start_time + reading_dt, "simple", state=reading_v)

    state0p1 = mf.tick(start_time + output_dt, readings=[reading1])

    print("state")
    print(state.data[0, 0])
    print("reading")
    print(reading_v)
    print("state0p1")
    print(state0p1.state)
    print("reading")
    print(reading_v)
    print("diff")
    print((state0p1.state.data) - (reading_v))
    assert np.isclose(state0p1.state.data, reading_v, atol=2.0e-8).all()


@settings(deadline=None)
@given(
    sampled_from(samples_dt_sec()),
    permutations(
        [
            ShuffleId.Zero,
            ShuffleId.Sensor0,
            ShuffleId.Sensor1,
            ShuffleId.Sensor2,
            ShuffleId.Sensor3,
            ShuffleId.Output,
        ]
    ),
)
def test_tick_multi_reading(output_dt, shuffle_order):
    output_dt, options = parse_options(output_dt, shuffle_order)
    start_time = 10.0
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map=calibration_map)
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    reading_v = -3.0
    readings = [StampedReading(t, "simple", state=reading_v) for t in options]

    state0p1 = mf.tick(start_time + output_dt, readings=readings)

    assert (state.data != state0p1.state.data).any()
