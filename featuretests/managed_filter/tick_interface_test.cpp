#include <featuretest/example.h>  // Generated
#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

#include <vector>

namespace testing {
using formak::runtime::ManagedFilter;

TEST(ManagedFilterTickTest, TimeOnly) {
  featuretest::State state(featuretest::StateOptions{.v = 1.0});
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      0.0, {
               .state = state,
               .covariance = {},
           });

  featuretest::Control control;

  auto state0p1 = mf.tick(0.1, control);

  auto state0p2 = mf.tick(0.2, control);

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

TEST(ManagedFilterTickTest, EmptyReadings) {
  featuretest::State state(featuretest::StateOptions{.v = 1.0});
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      1.0, {
               .state = state,
               .covariance = {},
           });

  featuretest::Control control;

  auto state0p1 = mf.tick(1.1, control, {});

  auto state0p2 = mf.tick(1.2, control, {});

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

TEST(ManagedFilterTickTest, OneReading) {
  featuretest::State state(featuretest::StateOptions{.v = 1.0});
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      2.0, {
               .state = state,
               .covariance = {},
           });

  featuretest::Control control;

  auto state0p1 = ([&mf, &control]() {
    featuretest::Simple reading{featuretest::SimpleOptions{}};
    return mf.tick(2.1, control, {mf.wrap(2.05, reading)});
  })();

  auto state0p2 = ([&mf, &control]() {
    featuretest::Simple reading{featuretest::SimpleOptions{}};
    return mf.tick(2.2, control, {mf.wrap(2.15, reading)});
  })();

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

TEST(ManagedFilterTickTest, MultipleReadings) {
  using namespace featuretest;
  State state(StateOptions{.v = 1.0});
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      3.0, {
               .state = state,
               .covariance = {},
           });

  Control control;

  auto state0p1 = ([&mf, &control]() {
    return mf.tick(3.1, control,
                   {
                       mf.wrap<Simple>(3.05, SimpleOptions{}),
                       mf.wrap<Simple>(3.06, SimpleOptions{}),
                       mf.wrap<Simple>(3.07, SimpleOptions{}),
                   });
  })();

  auto state0p2 = ([&mf, &control]() {
    return mf.tick(3.2, control,
                   {
                       mf.wrap<Simple>(3.15, SimpleOptions{}),
                       mf.wrap<Accel>(3.16, AccelOptions{.a = 1.0}),
                       mf.wrap<Simple>(3.17, SimpleOptions{}),
                   });
  })();

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

}  // namespace testing
