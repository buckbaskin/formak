import numpy as np
import pytest
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

    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

    ekf = python.compile_ekf(
        state_model=model,
        process_noise={thrust: 1.0},
        sensor_models={"simple": {ui.Symbol("v"): ui.Symbol("v")}},
        sensor_noises={"simple": np.eye(1)},
    )
    return ekf


def test_tick_time_only():
    ekf = make_ekf()
    state = np.array([[0.0, 0.0, 0.0, 0.0]]).transpose()
    covariance = np.eye(4)
    control = np.array([[0.0]])
    mf = ManagedFilter(ekf=ekf, start_time=0.0, state=state, covariance=covariance)

    state0p1 = mf.tick(0.1, control)

    state0p2 = mf.tick(0.2, control)

    assert np.any(state0p1 != state0p2)


def test_tick_empty_sensor_readings():
    ekf = make_ekf()
    state = np.array([[0.0, 1.0, 0.0, 0.0]]).transpose()
    covariance = np.eye(4)
    control = np.array([[0.0]])
    mf = ManagedFilter(ekf=ekf, start_time=2.0, state=state, covariance=covariance)

    state0p1 = mf.tick(0.1, control, [])

    state0p2 = mf.tick(0.2, control, [])

    assert np.any(state0p1 != state0p2)


def test_tick_one_sensor_reading():
    # featuretest::State state(featuretest::StateOptions{.v = 1.0});
    # formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
    #     2.0, {
    #              .state = state,
    #              .covariance = {},
    #          });

    # featuretest::Control control;

    # auto state0p1 = ([&mf, &control]() {
    #   featuretest::Simple reading{featuretest::SimpleOptions{}};
    #   return mf.tick(2.1, control, {mf.wrap(2.05, reading)});
    # })();

    # auto state0p2 = ([&mf, &control]() {
    #   featuretest::Simple reading{featuretest::SimpleOptions{}};
    #   return mf.tick(2.2, control, {mf.wrap(2.15, reading)});
    # })();

    # EXPECT_NE(state0p1.state.data, state0p2.state.data);
    ekf = make_ekf()
    state = np.array([[0.0, 1.0, 0.0, 0.0]]).transpose()
    covariance = np.eye(4)
    control = np.array([[0.0]])
    mf = ManagedFilter(ekf=ekf, start_time=2.0, state=state, covariance=covariance)

    reading1 = StampedReading(2.05, "simple", np.zeros((1,1)))

    state0p1 = mf.tick(0.1, control, [reading1])

    reading2 = StampedReading(2.15, "simple", np.zeros((1,1)))

    state0p2 = mf.tick(0.2, control, [reading2])

    assert np.any(state0p1 != state0p2)


def test_tick_multiple_sensor_reading():
    ekf = make_ekf()
    state = np.array([[0.0, 1.0, 0.0, 0.0]]).transpose()
    covariance = np.eye(4)
    control = np.array([[0.0]])
    mf = ManagedFilter(ekf=ekf, start_time=2.0, state=state, covariance=covariance)

    readings1 = [
        StampedReading(0.10, "simple", np.zeros((1,1))),
        StampedReading(0.10, "simple", np.zeros((1,1))),
        StampedReading(0.10, "simple", np.zeros((1,1))),
    ]

    current_time, (state0p1, _cov) = mf.tick(0.1, control, readings1)

    readings2 = [
        StampedReading(0.10, "simple", np.zeros((1,1))),
        StampedReading(0.10, "simple", np.zeros((1,1))),
        StampedReading(0.10, "simple", np.zeros((1,1))),
    ]

    current_time, (state0p2, _cov) = mf.tick(0.2, control, readings2)

    assert np.any(state0p1 != state0p2)
