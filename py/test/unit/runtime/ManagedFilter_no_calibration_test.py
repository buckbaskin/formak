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


def make_ekf():
    dt = ui.Symbol("dt")

    state = ui.Symbol("state")

    control_velocity = ui.Symbol("control_velocity")

    state_model = {state: state + dt * (control_velocity)}

    state_set = {state}
    control_set = {control_velocity}

    model = ui.Model(
        dt=dt,
        state=state_set,
        control=control_set,
        state_model=state_model,
    )

    ekf = python.compile_ekf(
        state_model=model,
        process_noise={control_velocity: 1.0},
        sensor_models={"simple": {state: state}},
        sensor_noises={"simple": {state: 1e-9}},
        config={"innovation_filtering": None},
    )
    return ekf


def test_constructor():
    ekf = make_ekf()
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)
    _mf = ManagedFilter(ekf=ekf, start_time=0.0, state=state, covariance=covariance)


@settings(deadline=None)
@given(sampled_from(samples_dt_sec()))
def test_tick_no_readings(dt):
    start_time = 10.0
    ekf = make_ekf()
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    control = ekf.Control(control_velocity=-1.0)
    state0p1 = mf.tick(start_time + dt, control=control)

    print("state")
    print(state0p1.state)
    print("reading")
    print(state, dt, control.data[0, 0])
    print(state.data + dt * control.data[0, 0])
    print("diff")
    print((state0p1.state.data) - (state.data + dt * control.data[0, 0]))
    assert np.isclose(
        state0p1.state.data, state.data + dt * control.data[0, 0], atol=2.0e-14
    ).all()
    if dt != 0.0:
        assert state0p1.covariance.data > covariance.data
    else:
        assert state0p1.covariance.data == covariance.data


@settings(deadline=None)
@given(sampled_from(samples_dt_sec()))
def test_tick_empty_readings(dt):
    start_time = 10.0
    ekf = make_ekf()
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    control = ekf.Control(control_velocity=-1.0)
    state0p1 = mf.tick(start_time + dt, control=control, readings=[])

    assert np.isclose(
        state0p1.state.data, state.data + dt * control.data[0, 0], atol=2.0e-14
    ).all()
    if dt != 0.0:
        assert state0p1.covariance.data > covariance.data
    else:
        assert state0p1.covariance.data == covariance.data


@settings(deadline=None)
@given(sampled_from(samples_dt_sec()), sampled_from(samples_dt_sec()))
def test_tick_one_reading(output_dt, reading_dt):
    start_time = 10.0
    ekf = make_ekf()
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    control = ekf.Control(control_velocity=-1.0)
    reading_v = -3.0
    reading1 = StampedReading(
        start_time + reading_dt, "simple", ekf.make_reading("simple", state=reading_v)
    )

    state0p1 = mf.tick(start_time + output_dt, control=control, readings=[reading1])

    dt = output_dt - reading_dt

    print("state")
    print(state.data[0, 0])
    print("reading")
    print(reading_v)
    print("state0p1")
    print(state0p1.state)
    print("reading")
    print(reading_v, control.data[0, 0], dt)
    print(reading_v + control.data[0, 0] * dt)
    print("diff")
    print((state0p1.state.data) - (reading_v + control.data[0, 0] * dt))
    assert np.isclose(
        state0p1.state.data, reading_v + control.data[0, 0] * dt, atol=2.0e-8
    ).all()


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
    ekf = make_ekf()
    state = ekf.State(state=4.0)
    covariance = ekf.Covariance(state=1.0)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    control = ekf.Control(control_velocity=-1.0)
    reading_v = -3.0
    readings = [
        StampedReading(t, "simple", ekf.make_reading("simple", state=reading_v))
        for t in options
    ]

    state0p1 = mf.tick(start_time + output_dt, control=control, readings=readings)

    assert (state.data != state0p1.state.data).any()
