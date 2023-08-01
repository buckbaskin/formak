import numpy as np
from formak.runtime import ManagedFilter, StampedReading
from hypothesis import given
from hypothesis.strategies import permutations, sampled_from

from formak import python, ui


def samples_dt_sec():
    return [0.0, 0.1, -0.1, -1.5, 2.7]


def make_ekf(calibration_map):
    dt = ui.Symbol("dt")

    state = ui.Symbol("state")

    control_velocity = ui.Symbol("control_velocity")
    calibration_velocity = ui.Symbol("calibration_velocity")

    state_model = {state: state + dt * (control_velocity + calibration_velocity)}

    state_set = {state}
    control_set = {control_velocity}

    model = ui.Model(
        dt=dt,
        state=state_set,
        control=control_set,
        state_model=state_model,
        calibration={calibration_velocity},
    )

    ekf = python.compile_ekf(
        state_model=model,
        process_noise={control_velocity: 1.0},
        sensor_models={"simple": {state: state}},
        sensor_noises={"simple": np.eye(1) * 1e-9},
        calibration_map=calibration_map,
    )
    return ekf


def test_constructor():
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map)
    state = np.array([[4.0]])
    covariance = np.array([[1.0]])
    _mf = ManagedFilter(ekf=ekf, start_time=0.0, state=state, covariance=covariance)


@given(sampled_from(samples_dt_sec()))
def test_tick_no_readings(dt):
    start_time = 10.0
    state = np.array([[4.0]])
    covariance = np.array([[1.0]])
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map=calibration_map)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    control = np.array([[-1.0]])
    state0p1 = mf.tick(start_time + dt, control)

    print("state")
    print(state0p1.state)
    print("reading")
    print(state, dt, control[0, 0])
    print(state + dt * control[0, 0])
    print("diff")
    print((state0p1.state) - (state + dt * control[0, 0]))
    assert np.isclose(state0p1.state, state + dt * control[0, 0], atol=2.0e-14).all()
    if dt != 0.0:
        assert state0p1.covariance > covariance
    else:
        assert state0p1.covariance == covariance


@given(sampled_from(samples_dt_sec()))
def test_tick_empty_readings(dt):
    start_time = 10.0
    state = np.array([[4.0]])
    covariance = np.array([[1.0]])
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map=calibration_map)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    control = np.array([[-1.0]])
    state0p1 = mf.tick(start_time + dt, control, [])

    assert np.isclose(state0p1.state, state + dt * control[0, 0], atol=2.0e-14).all()
    if dt != 0.0:
        assert state0p1.covariance > covariance
    else:
        assert state0p1.covariance == covariance


@given(sampled_from(samples_dt_sec()))
def test_tick_one_reading(dt):
    start_time = 10.0
    state = np.array([[4.0]])
    covariance = np.array([[1.0]])
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map=calibration_map)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    control = np.array([[-1.0]])
    reading_v = -3.0
    reading1 = StampedReading(2.05, "simple", np.array([[reading_v]]))

    state0p1 = mf.tick(start_time + dt, control, [reading1])

    assert np.isclose(state0p1.state, reading_v + control[0, 0] * dt, atol=2.0e-8).all()


# INSTANTIATE_TEST_SUITE_P(
#     TickTimings, ManagedFilterTest,
#     ::testing::Combine(::testing::Values(-1.5, -0.1, 0.0, 0.1, 2.7),
#                        ::testing::Values(-1.5, -0.1, 0.0, 0.1, 2.7)));

# namespace multitick {
# using test::tools::OrderOptions;
#
# class ManagedFilterMultiTest
#     : public ::testing::Test,
#       public ::testing::WithParamInterface<test::tools::OrderOptions> {};
#
# TEST_P(ManagedFilterMultiTest, TickMultiReading) {
#   using formak::runtime::ManagedFilter;
#   OrderOptions options = GetParam();
#
#   double start_time = 10.0;
#   StateAndVariance initial_state{
#       .state = 4.0,
#       .covariance = 1.0,
#   };
#   Calibration calibration{.velocity = 0.0};
#   ManagedFilter<TestImpl> mf(start_time, initial_state, calibration);
#
#   Control control{.velocity = -1.0};
#   double reading = -3.0;
#
#   std::vector<ManagedFilter<TestImpl>::StampedReading> one{
#       ManagedFilter<TestImpl>::wrap(start_time + options.sensor_dt[0],
#                                     Reading(reading)),
#       ManagedFilter<TestImpl>::wrap(start_time + options.sensor_dt[1],
#                                     Reading(reading)),
#       ManagedFilter<TestImpl>::wrap(start_time + options.sensor_dt[2],
#                                     Reading(reading)),
#       ManagedFilter<TestImpl>::wrap(start_time + options.sensor_dt[3],
#                                     Reading(reading)),
#   };
#
#   StateAndVariance next_state =
#       mf.tick(start_time + options.output_dt, control, one);
#
#   EXPECT_NE(next_state.state, initial_state.state);
# }


@given(sampled_from(samples_dt_sec()))
def test_tick_multi_reading(dt):
    start_time = 10.0
    state = np.array([[4.0]])
    covariance = np.array([[1.0]])
    calibration_map = {ui.Symbol("calibration_velocity"): 0.0}
    ekf = make_ekf(calibration_map=calibration_map)

    mf = ManagedFilter(
        ekf=ekf,
        start_time=start_time,
        state=state,
        covariance=covariance,
    )

    control = np.array([[-1.0]])
    reading_v = -3.0
    # TODO(buck): recreate multi-order readings
    assert False
    reading1 = StampedReading(2.05, "simple", np.array([[reading_v]]))

    state0p1 = mf.tick(start_time + dt, control, [reading1])

    assert np.isclose(
        state0p1.state, reading_v + control[0, 0] * dt, atol=2.0e-14
    ).all()
    if dt != 0.0:
        assert state0p1.covariance > covariance
    else:
        assert state0p1.covariance == covariance


#
# INSTANTIATE_TEST_SUITE_P(MultiTickTimings, ManagedFilterMultiTest,
#                          ::testing::ValuesIn(test::tools::AllOptions()));
# }  // namespace multitick
