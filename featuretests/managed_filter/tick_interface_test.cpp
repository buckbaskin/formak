#include <featuretest/example.h>  // Generated
#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

namespace testing {
using formak::runtime::ManagedFilter;

TEST(ManagedFilterTickTest, TimeOnly) {
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf;

  auto state0p1 = mf.tick(0.1);

  auto state0p2 = mf.tick(0.2);

  EXPECT_NE(state0p1, state0p2);
}

TEST(ManagedFilterTickTest, EmptyReadings) {
  [[maybe_unused]] ManagedFilter<featuretest::ExtendedKalmanFilter> mf;
}

TEST(ManagedFilterTickTest, OneReading) {
  [[maybe_unused]] ManagedFilter<featuretest::ExtendedKalmanFilter> mf;
}

TEST(ManagedFilterTickTest, MultipleReadings) {
  [[maybe_unused]] ManagedFilter<featuretest::ExtendedKalmanFilter> mf;
}

}  // namespace testing
