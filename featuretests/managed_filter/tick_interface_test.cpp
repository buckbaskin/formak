#include <featuretest/example.h>  // Generated
#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

#include <vector>

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

  featuretest::Control control;

  auto state0p1 = mf.tick(0.1, control, {});

  auto state0p2 = mf.tick(0.2, control, {});

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

TEST(ManagedFilterTickTest, OneReading) {
  ManagedFilter<featuretest::ExtendedKalmanFilter> mf;

  featuretest::Control control;

  auto state0p1 = ([&mf, &control]() {
    featuretest::Simple reading{featuretest::SimpleOptions{.timestamp = 0.05}};
    return mf.tick(0.1, control,
                   std::vector<featuretest::StampedReading>{reading});
  })();

  auto state0p2 = ([&mf, &control]() {
    featuretest::Simple reading{featuretest::SimpleOptions{.timestamp = 0.15}};
    return mf.tick(0.2, control,
                   std::vector<featuretest::StampedReading>{reading});
  })();

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

TEST(ManagedFilterTickTest, MultipleReadings) {
  [[maybe_unused]] ManagedFilter<featuretest::ExtendedKalmanFilter> mf;
}

}  // namespace testing
