#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

namespace unit {

struct StateAndVariance {
  double state = 0.0;
  double covariance = 1.0;
};
// TODO(buck): Do all filters have a control?
struct Control {};

struct StampedReadingBase;

struct TestImpl {
  using StateAndVarianceT = StateAndVariance;
  using ControlT = Control;
  using StampedReadingBaseT = StampedReadingBase;

  template <typename ReadingT>
  StateAndVariance sensor_model(const StateAndVarianceT& input,
                                const ReadingT& reading) const {
    return input;
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

TEST(ManagedFilter, Constructor) {
  // template <typename Impl>
  // class ManagedFilter {
  //  public:
  //   ManagedFilter(double initialTimestamp,
  //                 const typename Impl::StateAndVarianceT& initialState)
  //       : _currentTime(initialTimestamp), _state(initialState) {
  //   }
  // }

  // [[maybe_unused]] because this test is focused on the constructor only.
  // Passes if construction and deconstruction are successful
  [[maybe_unused]] formak::runtime::ManagedFilter<TestImpl> mf(
      1.23, StateAndVariance{.state = 4.0, .covariance = 1.0});

  // Note(buck): This test is helpful (compiler error because _impl isn't
  // initialized...)
}

// struct StampedReading {
//   double timestamp = 0.0;
//   std::shared_ptr<typename Impl::StampedReadingBaseT> data;
// };
// template <typename ReadingT>
// StampedReading wrap(double timestamp, const ReadingT& reading) const {
//   return StampedReading{
//       .timestamp = timestamp,
//       .data = std::shared_ptr<typename Impl::StampedReadingBaseT>(
//           new ReadingT(reading)),
//   };
// }
TEST(ManagedFilter, StampedReading) {
  using formak::runtime::ManagedFilter;

  double reading = 1.0;

  formak::runtime::ManagedFilter<TestImpl>::StampedReading stamped_reading =
      ManagedFilter<TestImpl>::wrap(5.0, Reading(reading));

  TestImpl impl;
  StateAndVariance state;

  EXPECT_DOUBLE_EQ(stamped_reading.timestamp, 5.0);
  EXPECT_DOUBLE_EQ(stamped_reading.data->sensor_model(impl, state).state,
                   reading);
}

//
// typename Impl::StateAndVarianceT tick(
//     double outputTime, const typename Impl::ControlT& control) {
//   return processUpdate(outputTime, control);
// }
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
