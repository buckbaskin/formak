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
    featuretest::Simple reading{featuretest::SimpleOptions{}};
    return mf.tick(0.1, control, {mf.wrap(0.05, reading)});
  })();

  auto state0p2 = ([&mf, &control]() {
    featuretest::Simple reading{featuretest::SimpleOptions{}};
    return mf.tick(0.2, control, {mf.wrap(0.15, reading)});
  })();

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

TEST(ManagedFilterTickTest, MultipleReadings) {
  using namespace featuretest;
  ManagedFilter<ExtendedKalmanFilter> mf;

  Control control;

  auto state0p1 = ([&mf, &control]() {
    return mf.tick(0.1, control,
                   {
                       mf.wrap<Simple>(0.05, SimpleOptions{}),
                       mf.wrap<Simple>(0.06, SimpleOptions{}),
                       mf.wrap<Simple>(0.07, SimpleOptions{}),
                   });
  })();

  auto state0p2 = ([&mf, &control]() {
    return mf.tick(0.2, control,
                   {
                       mf.wrap<Simple>(0.15, SimpleOptions{}),
                       mf.wrap<Accel>(0.16, AccelOptions{.a = 1.0}),
                       mf.wrap<Simple>(0.17, SimpleOptions{}),
                   });
  })();

  EXPECT_NE(state0p1.state.data, state0p2.state.data);
}

}  // namespace testing
