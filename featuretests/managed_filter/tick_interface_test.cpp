#include <featuretest/example.h>  // Generated
#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

namespace testing {
using formak::runtime::ManagedFilter;

TEST(ManagedFilterTickTest, TimeOnly) {
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf;

  featuretest::Control control;

  auto state0p1 = mf.tick(0.1, control);

  auto state0p2 = mf.tick(0.2, control);

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

TEST(ManagedFilterTickTest, EmptyReadings) {
  ManagedFilter<featuretest::ExtendedKalmanFilter> mf;
  // using ManagedFilter<featuretest::ExtendedKalmanFilter>::StampedReading;

  featuretest::Control control;

  auto state0p1 = mf.tick(0.1, control, {});

  auto state0p2 = mf.tick(0.2, control, {});

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

TEST(ManagedFilterTickTest, OneReading) {
  [[maybe_unused]] ManagedFilter<featuretest::ExtendedKalmanFilter> mf;
}

TEST(ManagedFilterTickTest, MultipleReadings) {
  [[maybe_unused]] ManagedFilter<featuretest::ExtendedKalmanFilter> mf;
}

}  // namespace testing
