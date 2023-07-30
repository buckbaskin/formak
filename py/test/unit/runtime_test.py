from formak.runtime import ManagedFilter

# TEST(ManagedFilterTest, Constructor) {
#   // [[maybe_unused]] because this test is focused on the constructor only.
#   // Passes if construction and deconstruction are successful
#   Calibration calibration{.velocity = 0.0};
#   StateAndVariance state_and_variance{.state = 4.0, .covariance = 1.0};
#   [[maybe_unused]] formak::runtime::ManagedFilter<TestImpl> mf(
#       1.23, state_and_variance, calibration);
# }
#
# TEST(ManagedFilterTest, StampedReading) {
#   using formak::runtime::ManagedFilter;
#
#   double reading = 1.0;
#
#   Reading message{reading};
#   ManagedFilter<TestImpl>::StampedReading stamped_reading =
#       ManagedFilter<TestImpl>::wrap(5.0, message);
#
#   TestImpl impl;
#   StateAndVariance state;
#
#   EXPECT_DOUBLE_EQ(stamped_reading.timestamp, 5.0);
#   // Can't directly address the .reading member of the child type. Instead, use
#   // the sensor_model interface to access (by stuffing into the .state member of
#   // the output)
#   Calibration calibration{.velocity = 0.0};
#   EXPECT_DOUBLE_EQ(
#       stamped_reading.data->sensor_model(impl, state, calibration).state,
#       message.reading);
# }
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


def test_constructor():
    ekf = make_ekf()
    state = np.array([[0.0, 1.0, 0.0, 0.0]]).transpose()
    covariance = np.eye(4)
    mf = ManagedFilter(ekf=ekf, start_time=0.0, state=state, covariance=covariance)

def test_tick_no_readings():
    ekf = make_ekf()
    start_time = 10.0
    state = 4.0
    covariance = 1.0
    calibration_map = {ui.Symbol('velocity'): 0.0}

    mf = ManagedFilter(ekf=ekf, start_time=start_time, state=state, covariance=covariance, calibration_map=calibration_map)

    control = np.array([[-1.0]])
    # TODO(buck): try this for positive and negative dt
    dt = 0.1
    state0p1 = mf.tick(start_time + dt, control)

    assert np.isclose(state0p1.state, state + dt * control[0,0], atol=2.0e-14)
    if dt != 0.0:
        assert next_state.covariance > covariance
    else:
        assert next_state.covariance == covariance

# TEST_P(ManagedFilterTest, TickNoReadings) {
#   using formak::runtime::ManagedFilter;
#   Options options(GetParam());
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
#
#   double dt = options.output_dt;
#   StateAndVariance next_state = mf.tick(start_time + dt, control);
#
#   EXPECT_NEAR(next_state.state, initial_state.state + dt * control.velocity,
#               2.0e-14)
#       << "  diff: "
#       << (next_state.state - (initial_state.state + dt * control.velocity));
#   if (options.output_dt != 0.0) {
#     EXPECT_GT(next_state.covariance, initial_state.covariance);
#   } else {
#     EXPECT_DOUBLE_EQ(next_state.covariance, initial_state.covariance);
#   }
# }
