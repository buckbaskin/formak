#include <featuretest/nonlinear.h>
#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

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

TEST(InnovationFilteringTest, ObviousInnovationRejections) {
  // ekf, compass_model = make_ekf()
  // # Note: state = heading, x, y
  // state = np.array([[0.0, 1.0, 0.0]]).transpose()
  // covariance = np.eye(3)
  featuretest::State state({
      .heading = 0.0,
      .x = 1.0,
      .y = 0.0,
  });

  // control = np.array([[0.0, 1.0]]).transpose()
  featuretest::Control control({
      ._heading_err = 1.0,
      .velocity = 0.0,
  });

  // mf = runtime.ManagedFilter(
  //     ekf=ekf, start_time=0.0, state=state, covariance=covariance
  // )
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      0.0, {
               .state = state,
               .covariance = {},
           });

  // readings = np.random.default_rng(seed=3).normal(
  //     loc=0.0, scale=TRUE_SCALE / 2.0, size=(100,)
  // )
  std::vector<double> readings;

  ASSERT_GT(readings.size(), 0);
  // readings[readings.shape[0] // 4] = radians(180.0)
  // readings[readings.shape[0] // 3] = radians(180.0)
  // readings[readings.shape[0] // 2] = radians(180.0)
  readings[readings.size() / 4] = radians(180.0);
  readings[readings.size() / 3] = radians(180.0);
  readings[readings.size() / 2] = radians(180.0);

  // TODO(buck): for loop
  // for idx, r in enumerate(readings):
  //     s = mf.tick(
  //         0.1 * idx,
  //         control=control,
  //         readings=[
  //             runtime.StampedReading(0.1 * idx - 0.05, "compass",
  //             np.array([[r]]))
  //         ],
  //     )
  for (size_t idx = 0; idx < readings.size(); ++idx) {
    auto r = readings[idx];
    auto s = mf.tick(0.1, control);

    // TODO(buck): assertions in for loop
    //     if abs(compass_model(s.state)) >= TRUE_SCALE * 4:
    //         print({"idx": idx, "reading": degrees(r)})
    //     assert abs(compass_model(s.state)) < TRUE_SCALE * 4
    ASSERT_LT(std::abs(s.state.heading()), TRUE_SCALE * 4)
        << "idx: " << idx << " reading: " << degrees(r);
  }

  FAIL() << "Not Implemented";
}

}  // namespace featuretest
