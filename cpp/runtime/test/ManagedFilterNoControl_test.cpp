#include <formak/runtime/ManagedFilter.h>
#include <formak/runtime/test/tools.h>
#include <gtest/gtest.h>

#include <iterator>  // back_inserter
#include <tuple>
#include <vector>

namespace unit {

namespace {
struct StateAndVariance {
  double state = 0.0;
  double covariance = 1.0;
};
struct Calibration {
  double velocity = 0.0;
};

struct StampedReadingBase;

// TODO(buck): Provide a C++ struct of known format as a constexpr member of the
// EKFImpl so that these CalibrationT, max_dt_sec, etc can be looked up in a
// known format
struct TestImpl {
  struct Tag {
   private:
    class Key {};

   public:
    using StateAndVarianceT = StateAndVariance;
    using CalibrationT = Calibration;
    using ControlT = Key;
    using StampedReadingBaseT = StampedReadingBase;
    static constexpr double max_dt_sec = 0.05;
    static constexpr bool enable_calibration = true;
    static constexpr bool enable_control = false;
  };

  template <typename ReadingT>
  StateAndVariance sensor_model(const StateAndVariance& input,
                                const Calibration& calibration,
                                const ReadingT& reading) const {
    return StateAndVariance{.state = reading.reading, .covariance = 1.0};
  }

  StateAndVariance process_model(double dt, const StateAndVariance& state,
                                 const Calibration& calibration) const {
    return {
        .state = state.state + dt * calibration.velocity,
        .covariance = state.covariance + std::abs(dt),
    };
  }
};

struct StampedReadingBase {
  virtual StateAndVariance sensor_model(
      const TestImpl& impl, const StateAndVariance& input,
      const Calibration& calibration) const = 0;
};
struct Reading : public StampedReadingBase {
  Reading(double reading_) : StampedReadingBase(), reading(reading_) {
  }

  StateAndVariance sensor_model(const TestImpl& impl,
                                const StateAndVariance& input,
                                const Calibration& calibration) const override {
    return impl.sensor_model(input, calibration, *this);
  }

  double reading = 0.0;
};
}  // namespace

TEST(ManagedFilterTest, Constructor) {
  // [[maybe_unused]] because this test is focused on the constructor only.
  // Passes if construction and deconstruction are successful
  [[maybe_unused]] formak::runtime::ManagedFilter<TestImpl> mf(
      1.23, StateAndVariance{.state = 4.0, .covariance = 1.0});
}

TEST(ManagedFilterTest, StampedReading) {
  using formak::runtime::ManagedFilter;

  double reading = 1.0;

  ManagedFilter<TestImpl>::StampedReading stamped_reading =
      ManagedFilter<TestImpl>::wrap(5.0, Reading(reading));

  TestImpl impl;
  StateAndVariance state;

  EXPECT_DOUBLE_EQ(stamped_reading.timestamp, 5.0);
  // Can't directly address the .reading member of the child type. Instead, use
  // the sensor_model interface to access (by stuffing into the .state member of
  // the output)
  Calibration calibration{.velocity = -1.0};
  EXPECT_DOUBLE_EQ(
      stamped_reading.data->sensor_model(impl, state, calibration).state,
      reading);
}

namespace tick {
struct Options {
  Options(const std::tuple<double, double>& generator_result)
      : output_dt(std::get<0>(generator_result)),
        reading_dt(std::get<1>(generator_result)) {
  }

  double output_dt;
  double reading_dt;
};
std::ostream& operator<<(std::ostream& o, const Options& options) {
  o << "Options{.output_dt=" << options.output_dt
    << ", .reading_dt=" << options.reading_dt << "}";
  return o;
}
class ManagedFilterTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<double, double>> {};

TEST_P(ManagedFilterTest, TickNoReadings) {
  using formak::runtime::ManagedFilter;
  Options options(GetParam());

  double start_time = 10.0;
  StateAndVariance initial_state{
      .state = 4.0,
      .covariance = 1.0,
  };
  Calibration calibration{.velocity = -1.0};
  ManagedFilter<TestImpl> mf(start_time, initial_state, calibration);

  double dt = options.output_dt;
  StateAndVariance next_state = mf.tick(start_time + dt);

  EXPECT_NEAR(next_state.state, initial_state.state + dt * calibration.velocity,
              2.0e-14)
      << "  diff: "
      << (next_state.state - (initial_state.state + dt * calibration.velocity));
  if (options.output_dt != 0.0) {
    EXPECT_GT(next_state.covariance, initial_state.covariance);
  } else {
    EXPECT_DOUBLE_EQ(next_state.covariance, initial_state.covariance);
  }
}

TEST_P(ManagedFilterTest, TickEmptyReadings) {
  using formak::runtime::ManagedFilter;
  Options options(GetParam());

  double start_time = 10.0;
  StateAndVariance initial_state{
      .state = 4.0,
      .covariance = 1.0,
  };
  Calibration calibration{.velocity = -1.0};
  ManagedFilter<TestImpl> mf(start_time, initial_state, calibration);

  std::vector<ManagedFilter<TestImpl>::StampedReading> empty;

  double dt = options.output_dt;
  StateAndVariance next_state = mf.tick(start_time + dt, empty);

  EXPECT_NEAR(next_state.state, initial_state.state + dt * calibration.velocity,
              1.5e-14)
      << "  diff: "
      << (next_state.state - (initial_state.state + dt * calibration.velocity));
  if (options.output_dt != 0.0) {
    EXPECT_GT(next_state.covariance, initial_state.covariance);
  } else {
    EXPECT_DOUBLE_EQ(next_state.covariance, initial_state.covariance);
  }
}

TEST_P(ManagedFilterTest, TickOneReading) {
  using formak::runtime::ManagedFilter;
  Options options(GetParam());

  double start_time = 10.0;
  StateAndVariance initial_state{
      .state = 4.0,
      .covariance = 1.0,
  };
  Calibration calibration{.velocity = -1.0};
  ManagedFilter<TestImpl> mf(start_time, initial_state, calibration);

  double reading = -3.0;

  std::vector<ManagedFilter<TestImpl>::StampedReading> one{
      ManagedFilter<TestImpl>::wrap(start_time + options.reading_dt,
                                    Reading(reading))};

  StateAndVariance next_state = mf.tick(start_time + options.output_dt, one);

  double dt = options.output_dt - options.reading_dt;
  EXPECT_NEAR(next_state.state, reading + calibration.velocity * dt, 2e-14)
      << "  diff: "
      << (next_state.state - (reading + dt * calibration.velocity));
}

INSTANTIATE_TEST_SUITE_P(
    TickTimings, ManagedFilterTest,
    ::testing::Combine(::testing::Values(-1.5, -0.1, 0.0, 0.1, 2.7),
                       ::testing::Values(-1.5, -0.1, 0.0, 0.1, 2.7)));
}  // namespace tick

namespace multitick {
using test::tools::OrderOptions;

class ManagedFilterMultiTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<test::tools::OrderOptions> {};

TEST_P(ManagedFilterMultiTest, TickMultiReading) {
  using formak::runtime::ManagedFilter;
  OrderOptions options = GetParam();

  double start_time = 10.0;
  StateAndVariance initial_state{
      .state = 4.0,
      .covariance = 1.0,
  };
  Calibration calibration{.velocity = -1.0};
  ManagedFilter<TestImpl> mf(start_time, initial_state, calibration);

  double reading = -3.0;

  std::vector<ManagedFilter<TestImpl>::StampedReading> one{
      ManagedFilter<TestImpl>::wrap(start_time + options.sensor_dt[0],
                                    Reading(reading)),
      ManagedFilter<TestImpl>::wrap(start_time + options.sensor_dt[1],
                                    Reading(reading)),
      ManagedFilter<TestImpl>::wrap(start_time + options.sensor_dt[2],
                                    Reading(reading)),
      ManagedFilter<TestImpl>::wrap(start_time + options.sensor_dt[3],
                                    Reading(reading)),
  };

  StateAndVariance next_state = mf.tick(start_time + options.output_dt, one);

  EXPECT_NE(next_state.state, initial_state.state);
}

INSTANTIATE_TEST_SUITE_P(MultiTickTimings, ManagedFilterMultiTest,
                         ::testing::ValuesIn(test::tools::AllOptions()));
}  // namespace multitick

}  // namespace unit
