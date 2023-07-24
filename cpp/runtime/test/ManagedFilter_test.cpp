#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

#include <vector>

namespace unit {

namespace {
struct StateAndVariance {
  double state = 0.0;
  double covariance = 1.0;
};
// TODO(buck): Do all filters have a control?
struct Control {
  double velocity = 0.0;
};

struct StampedReadingBase;

// Note: ManagedFilter assumes the Impl can be default constructed
struct TestImpl {
  using StateAndVarianceT = StateAndVariance;
  using ControlT = Control;
  using StampedReadingBaseT = StampedReadingBase;

  template <typename ReadingT>
  StateAndVariance sensor_model(const StateAndVarianceT& input,
                                const ReadingT& reading) const {
    return input;
  }

  StateAndVariance process_model(double dt, const StateAndVariance& state,
                                 const ControlT& control) const {
    return {
        .state = state.state + dt * control.velocity,
        .covariance = state.covariance + dt,
    };
  }
};

struct StampedReadingBase {
  virtual StateAndVariance sensor_model(
      const TestImpl& impl, const StateAndVariance& input) const = 0;
};
struct Reading : public StampedReadingBase {
  // TODO(buck): Am I forgetting to call the StampedReadingBase constructor in
  // my generated reading types?
  Reading(double reading_) : StampedReadingBase(), reading(reading_) {
  }

  StateAndVariance sensor_model(const TestImpl& impl,
                                const StateAndVariance& input) const override {
    return StateAndVariance{.state = reading, .covariance = 1.0};
  }

  double reading = 0.0;
};
}  // namespace

TEST(ManagedFilterTest, Constructor) {
  // [[maybe_unused]] because this test is focused on the constructor only.
  // Passes if construction and deconstruction are successful
  [[maybe_unused]] formak::runtime::ManagedFilter<TestImpl> mf(
      1.23, StateAndVariance{.state = 4.0, .covariance = 1.0});

  // Note(buck): This test is helpful (found a compiler error because _impl
  // wasn't initialized...)
}

TEST(ManagedFilterTest, StampedReading) {
  using formak::runtime::ManagedFilter;

  double reading = 1.0;

  formak::runtime::ManagedFilter<TestImpl>::StampedReading stamped_reading =
      ManagedFilter<TestImpl>::wrap(5.0, Reading(reading));

  TestImpl impl;
  StateAndVariance state;

  EXPECT_DOUBLE_EQ(stamped_reading.timestamp, 5.0);
  // Can't directly address the .reading member of the child type. Instead, use
  // the sensor_model interface to access (by stuffing into the .state member of
  // the output)
  EXPECT_DOUBLE_EQ(stamped_reading.data->sensor_model(impl, state).state,
                   reading);
}

// typename Impl::StateAndVarianceT tick(
//     double outputTime, const typename Impl::ControlT& control) {
//   return processUpdate(outputTime, control);
// }
namespace tick {
struct Options {
  double dt;
};
std::ostream& operator<<(std::ostream& o, const Options& options) {
  o << "Options{.dt=" << options.dt << "}";
  return o;
}
class ManagedFilterTest : public ::testing::Test,
                          public ::testing::WithParamInterface<Options> {};

TEST_P(ManagedFilterTest, TickNoReadings) {
  using formak::runtime::ManagedFilter;

  double start_time = 10.0;
  StateAndVariance initial_state{
      .state = 4.0,
      .covariance = 1.0,
  };
  ManagedFilter<TestImpl> mf(start_time, initial_state);

  Control control{.velocity = -1.0};

  double dt = GetParam().dt;
  StateAndVariance next_state = mf.tick(start_time + dt, control);

  EXPECT_DOUBLE_EQ(next_state.state, 4.0 + dt * control.velocity);
  if (GetParam().dt != 0.0) {
    EXPECT_GT(next_state.covariance, initial_state.covariance);
  }
}

INSTANTIATE_TEST_SUITE_P(TickTimings, ManagedFilterTest,
                         ::testing::Values(Options{-1.5}, Options{-0.1},
                                           Options{0.0}, Options{0.1},
                                           Options{2.7}));
}  // namespace tick

// typename Impl::StateAndVarianceT tick(
//     double outputTime, const typename Impl::ControlT& control,
//     const std::vector<StampedReading>& readings) {
//   for (const auto& stampedReading : readings) {
//     _state = processUpdate(stampedReading.timestamp, control);
//     _currentTime = stampedReading.timestamp;
//
//     _state = stampedReading.data->sensor_model(_impl, _state);
//   }
//
//   return tick(outputTime, control);
// }

}  // namespace unit
