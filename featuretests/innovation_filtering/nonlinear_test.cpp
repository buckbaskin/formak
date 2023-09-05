/// Innovation Filtering Feature Test
///
/// Create a model with states:
/// - x, y, heading, velocity model
///
/// Provide heading readings, expect rejecting 180 degree heading errors.
/// Nonlinear model provides clear divergence signal. If innovation filtering
/// isn't working as expected, then the model will flip into the wrong
/// direction.
///
/// Passes if the model rejects the high innovation updates.

#include <featuretest/nonlinear.h>
#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

#include <random>

namespace featuretest {

namespace {
constexpr double radians(double degrees) {
  // Precision isn't important, but probably should be changed for a std math
  // operation
  return degrees / 180.0 * 3.14159;
}
constexpr double degrees(double radians) {
  return radians * 180.0 / 3.14159;
}

constexpr double TRUE_SCALE = radians(5.0);
}  // namespace

std::vector<double> rng(size_t size, size_t seed = 1) {
  std::minstd_rand prng(seed);
  std::normal_distribution<> dist(0, TRUE_SCALE);

  std::vector<double> result;

  for (size_t i = 0; i < size; i++) {
    result.push_back(dist(prng));
  }

  return result;
}

TEST(InnovationFilteringTest, ObviousInnovationRejections) {
  featuretest::State state({
      .heading = 0.0,
      .x = 1.0,
      .y = 0.0,
  });
  ASSERT_DOUBLE_EQ(state.heading(), 0.0);

  featuretest::Control control({
      ._heading_err = 0.0,
      .velocity = 1.0,
  });

  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      0.0, {
               .state = state,
               .covariance = {},
           });

  std::vector<double> readings = rng(100, 3);

  ASSERT_EQ(readings.size(), 100);
  readings[readings.size() / 4] = radians(180.0);
  readings[readings.size() / 3] = radians(180.0);
  readings[readings.size() / 2] = radians(180.0);

  for (size_t idx = 0; idx < readings.size(); ++idx) {
    auto r = readings[idx];
    auto s = mf.tick(
        idx * 0.1, control,
        {
            mf.wrap<Compass>(idx * 0.1 - 0.05, CompassOptions{.heading = r}),
        });

    ASSERT_LT(std::abs(s.state.heading()), TRUE_SCALE * 4)
        << "idx: " << idx << " reading: " << degrees(r);
  }
}

}  // namespace featuretest
