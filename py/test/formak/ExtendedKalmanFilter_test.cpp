#include <formak/testing/stats.h>
#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>
#include <unit/simple-ekf.h>  // Generated

namespace integration {

namespace ekf_process_property_test {
using formak::testing::stats::MultivariateNormal;

RC_GTEST_PROP(CppModel, EKF_process_property, (double x, double y, double a)) {
  // def test_EKF_process_property(state_x, state_y, control_a):
  unit::ExtendedKalmanFilter ekf;
  double dt = 0.1;

  unit::Control control({a});
  unit::State state({x, y});
  unit::Covariance covariance;

  RC_PRE(std::isfinite(x) && std::isfinite(y) && std::isfinite(a));
  RC_PRE(std::abs(x) < 1e100 && std::abs(y) < 1e100 && std::abs(a) < 1e100);

  auto next = ekf.process_model(dt, {state, covariance}, control);

  RC_ASSERT(next.state.x() == x * y);
  RC_ASSERT(next.state.y() == y + a * dt);

  // try
  double starting_central_probability =
      MultivariateNormal(unit::State{}, covariance).pdf(unit::State{});
  double ending_central_probability =
      MultivariateNormal(unit::State{}, next.covariance).pdf(unit::State{});
  //     except np.linalg.LinAlgError:

  std::cout << "next.covariance" << std::endl;
  std::cout << next.covariance.data << std::endl;

  RC_ASSERT(ending_central_probability < starting_central_probability);
}

struct Options {
  double x;
  double y;
  double a;
};

class CppModelFailureCasesProcess
    : public ::testing::Test,
      public ::testing::WithParamInterface<Options> {};

TEST_P(CppModelFailureCasesProcess, RerunCases) {
  double a = GetParam().a;
  double x = GetParam().x;
  double y = GetParam().y;

  unit::ExtendedKalmanFilter ekf;
  double dt = 0.1;

  unit::Control control({a});
  unit::State state({x, y});
  unit::Covariance covariance;

  // Don't need pre for hand-inspected test cases
  // RC_PRE(std::isfinite(x) && std::isfinite(y) && std::isfinite(a));
  // RC_PRE(std::abs(x) < 1e100 && std::abs(y) < 1e100 && std::abs(a) < 1e100);

  auto next = ekf.process_model(dt, {state, covariance}, control);

  EXPECT_EQ(next.state.x(), x * y);
  EXPECT_EQ(next.state.y(), y + a * dt);

  // try
  double starting_central_probability =
      MultivariateNormal(unit::State{}, covariance).pdf(unit::State{});
  double ending_central_probability =
      MultivariateNormal(unit::State{}, next.covariance).pdf(unit::State{});
  //     except np.linalg.LinAlgError:

  std::cout << "next.covariance" << std::endl;
  std::cout << next.covariance.data << std::endl;

  EXPECT_LT(ending_central_probability, starting_central_probability);
}

INSTANTIATE_TEST_SUITE_P(PreviousFailureCases, CppModelFailureCasesProcess,
                         ::testing::Values(Options{0.0, 0.0, 0.0}));

}  // namespace ekf_process_property_test

namespace ekf_sensor_property_test {
using formak::testing::stats::MultivariateNormal;

RC_GTEST_PROP(CppModel, EKF_sensor_property, (double x, double y)) {
  // def test_EKF_sensor_property(x, y, a):
  unit::ExtendedKalmanFilter ekf;

  unit::State state({x, y});
  unit::Covariance covariance;

  RC_PRE(std::isfinite(x) && std::isfinite(y));
  RC_PRE(std::abs(x) < 1e100 && std::abs(y) < 1e100);

  double reading = 1.0;
  unit::SensorReading<(unit::SensorId::SIMPLE), unit::Simple> simple_reading{
      unit::Simple({reading})};
  auto next = ekf.sensor_model({state, covariance}, simple_reading);

  auto maybe_first_innovation =
      ekf.innovations<unit::SensorId::SIMPLE, unit::Simple>();
  RC_ASSERT(maybe_first_innovation.has_value());
  typename unit::Simple::InnovationT first_innovation =
      maybe_first_innovation.value();

  // try
  double starting_central_probability =
      MultivariateNormal(unit::State{}, covariance).pdf(unit::State{});
  double ending_central_probability =
      MultivariateNormal(unit::State{}, next.covariance).pdf(unit::State{});
  //     except np.linalg.LinAlgError:

  RC_ASSERT(ending_central_probability > starting_central_probability);

  // Run this to get a second innovation
  [[maybe_unused]] auto next2 = ekf.sensor_model(next, simple_reading);

  auto maybe_second_innovation =
      ekf.innovations<unit::SensorId::SIMPLE, unit::Simple>();
  RC_ASSERT(maybe_second_innovation.has_value());
  typename unit::Simple::InnovationT second_innovation =
      maybe_second_innovation.value();

  RC_ASSERT(first_innovation.norm() > second_innovation.norm() ||
            first_innovation.norm() == 0.0);
}

struct Options {
  double x;
  double y;
};

class CppModelFailureCasesSensor
    : public ::testing::Test,
      public ::testing::WithParamInterface<Options> {};

TEST_P(CppModelFailureCasesSensor, RerunCases) {
  double x = GetParam().x;
  double y = GetParam().y;

  // def test_EKF_sensor_property(x, y, a):
  unit::ExtendedKalmanFilter ekf;

  unit::State state({x, y});
  unit::Covariance covariance;

  // Don't need pre for hand-inspected test cases
  // RC_PRE(std::isfinite(x) && std::isfinite(y));
  // RC_PRE(std::abs(x) < 1e100 && std::abs(y) < 1e100);

  double reading = 1.0;
  unit::SensorReading<(unit::SensorId::SIMPLE), unit::Simple> simple_reading{
      unit::Simple({reading})};
  auto next = ekf.sensor_model({state, covariance}, simple_reading);

  auto maybe_first_innovation =
      ekf.innovations<unit::SensorId::SIMPLE, unit::Simple>();
  ASSERT_TRUE(maybe_first_innovation.has_value());
  typename unit::Simple::InnovationT first_innovation =
      maybe_first_innovation.value();

  // try
  double starting_central_probability =
      MultivariateNormal(unit::State{}, covariance).pdf(unit::State{});
  double ending_central_probability =
      MultivariateNormal(unit::State{}, next.covariance).pdf(unit::State{});
  //     except np.linalg.LinAlgError:

  RC_ASSERT(ending_central_probability > starting_central_probability);

  // Run this to get a second innovation
  [[maybe_unused]] auto next2 = ekf.sensor_model(next, simple_reading);

  auto maybe_second_innovation =
      ekf.innovations<unit::SensorId::SIMPLE, unit::Simple>();
  ASSERT_TRUE(maybe_second_innovation.has_value());
  typename unit::Simple::InnovationT second_innovation =
      maybe_second_innovation.value();

  EXPECT_TRUE(first_innovation.norm() > second_innovation.norm() ||
              first_innovation.norm() == 0.0);
}

INSTANTIATE_TEST_SUITE_P(PreviousFailureCases, CppModelFailureCasesSensor,
                         ::testing::Values(Options{0.0, 0.0}));

}  // namespace ekf_sensor_property_test

}  // namespace integration
