import numpy as np
import pytest

from formak import python, runtime, ui


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

    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

    ekf = python.compile_ekf(
        state_model=model,
        process_noise={thrust: 1.0},
        sensor_models={"simple": {ui.Symbol("v"): ui.Symbol("v")}},
        sensor_noises={"simple": np.eye(1)},
    )
    return ekf


def test_tick_time_only():
    # featuretest::State state(featuretest::StateOptions{.v = 1.0});
    # formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
    #     0.0, {
    #              .state = state,
    #              .covariance = {},
    #          });

    # featuretest::Control control;

    # auto state0p1 = mf.tick(0.1, control);

    # auto state0p2 = mf.tick(0.2, control);

    # EXPECT_NE(state0p1.state.data, state0p2.state.data);
    ekf = make_ekf()
    state = np.array([[0.0, 0.0, 0.0, 0.0]]).transpose()
    covariance = np.eye(4)
    mf = runtime.ManagedFilter(
        ekf=ekf, start_time=0.0, state=state, covariance=covariance
    )

    control = np.array([[0.0]])

    state0p1 = mf.tick(0.1, control)

    state0p2 = mf.tick(0.2, control)

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
