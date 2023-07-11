#include <gtest/gtest.h>
#include <unit/simple-ekf.h>  // Generated

namespace unit {

TEST(EKF, StateDefaultInitializationIsZero) {
  State state;
  EXPECT_EQ(state.data.rows(), 2);
  EXPECT_EQ(state.data.cols(), 1);

  EXPECT_EQ(state.x(), 0.0);
  EXPECT_EQ(state.y(), 0.0);
}

TEST(EKF, CovarianceDefaultInitializationIsIdentity) {
  Covariance covariance;
  EXPECT_EQ(covariance.data.rows(), 2);
  EXPECT_EQ(covariance.data.cols(), 2);

  EXPECT_EQ(covariance.x(), 1.0);
  EXPECT_EQ(covariance.y(), 1.0);

  EXPECT_EQ(covariance.data(1, 0), 0.0);
  EXPECT_EQ(covariance.data(0, 1), 0.0);
}

TEST(EKF, ControlConstructor) {
  {
    Control control;
    EXPECT_DOUBLE_EQ(control.a(), 0.0);
  }

  {
    Control control({0.2});
    EXPECT_DOUBLE_EQ(control.a(), 0.2);
  }
}

TEST(EKF, StateConstructor) {
  {
    State state;
    EXPECT_DOUBLE_EQ(state.x(), 0.0);
    EXPECT_DOUBLE_EQ(state.y(), 0.0);
  }

  {
    State state({0.2, -1.6});
    EXPECT_DOUBLE_EQ(state.x(), 0.2);
    EXPECT_DOUBLE_EQ(state.y(), -1.6);
  }
}

namespace process_with_control_test {
struct Options {
  double xStart;
  double yStart;

  double xEnd;
  double yEnd;
};

class ProcessWithControlTest : public ::testing::Test,
                               public ::testing::WithParamInterface<Options> {};

TEST_P(ProcessWithControlTest, Test) {
  ExtendedKalmanFilter ekf;
  double dt = 0.1;

  Control control({0.2});

  Covariance covariance;

  State state({GetParam().xStart, GetParam().yStart});

  auto next = ekf.process_model(dt, {state, covariance}, control);

  EXPECT_DOUBLE_EQ(next.state.x(), GetParam().xEnd);
  EXPECT_DOUBLE_EQ(next.state.y(), GetParam().yEnd);
}

INSTANTIATE_TEST_SUITE_P(StateTestCases, ProcessWithControlTest,
                         ::testing::Values(Options{0.0, 0.0, 0.0, 0.02},
                                           Options{0.0, 1.0, 0.1, 1.02},
                                           Options{1.0, 0.0, 1.0, 0.02},
                                           Options{1.0, 1.0, 1.1, 1.02}));
}  // namespace process_with_control_test

namespace ekf_sensor_test {
TEST(EKF, SensorSimple) {
  ExtendedKalmanFilter ekf;

  Covariance covariance;
  State state({0.0, 0.0});

  double reading = 1.0;

  Simple simple_reading({reading});
  EXPECT_EQ(simple_reading.data.rows(), 1);
  EXPECT_EQ(simple_reading.data.cols(), 1);

  auto next = ekf.sensor_model({state, covariance}, simple_reading);

  EXPECT_LT(abs(reading - next.state.data(0, 0)),
            abs(reading - state.data(0, 0)));
}
TEST(EKF, SensorCombined) {
  ExtendedKalmanFilter ekf;

  Covariance covariance;
  State state({0.0, 0.0});

  double reading = 1.0;

  Combined combined({reading});
  EXPECT_EQ(combined.data.rows(), 1);
  EXPECT_EQ(combined.data.cols(), 1);

  auto next = ekf.sensor_model({state, covariance}, combined);

  EXPECT_LT(abs(reading - next.state.data(0, 0)),
            abs(reading - state.data(0, 0)));
  EXPECT_LT(abs(reading - next.state.data(1, 0)),
            abs(reading - state.data(1, 0)));
}
}  // namespace ekf_sensor_test

TEST(EKF, SensorModelDetail) {
  double x = -1.0;
  double y = 2.0;
  State state({x, y});
  Covariance covariance;

  {
    Simple simple_reading{};
    Simple predicted =
        Simple::SensorModel::model({state, covariance}, simple_reading);
    EXPECT_DOUBLE_EQ(predicted.reading1(), x);
  }

  {
    Combined combined_reading{};
    Combined predicted =
        Combined::SensorModel::model({state, covariance}, combined_reading);
    EXPECT_DOUBLE_EQ(predicted.reading2(), x + y);
  }
}

TEST(EKF, SensorCovariances) {
  double x = -1.0;
  double y = 2.0;
  State state({x, y});
  Covariance covariance;

  {
    Simple simple_reading{};
    Simple::CovarianceT out =
        Simple::SensorModel::covariance({state, covariance}, simple_reading);
    EXPECT_DOUBLE_EQ(out(0, 0), 1.0);
  }

  {
    Combined combined_reading{};
    Combined::CovarianceT out = Combined::SensorModel::covariance(
        {state, covariance}, combined_reading);
    EXPECT_DOUBLE_EQ(out(0, 0), 4.0);
  }
}

TEST(EKF, SensorJacobian) {
  double x = -1.0;
  double y = 2.0;
  State state({x, y});
  Covariance covariance;

  {
    Simple simple_reading{};
    Simple::SensorJacobianT out =
        Simple::SensorModel::jacobian({state, covariance}, simple_reading);
    EXPECT_DOUBLE_EQ(out(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(out(0, 1), 0.0);
  }

  {
    Combined combined_reading{};
    Combined::SensorJacobianT out =
        Combined::SensorModel::jacobian({state, covariance}, combined_reading);
    EXPECT_DOUBLE_EQ(out(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(out(0, 1), 1.0);
  }
}

TEST(EKF, ProcessJacobian) {
  ExtendedKalmanFilter ekf;

  double dt = 0.05;

  double x = -1.0;
  double y = 2.0;

  State state({x, y});
  // Assumes default initialization of covariance (identity) and control (0)
  Covariance covariance;
  Control control;

  StateAndVariance next = ekf.process_model(dt, {state, covariance}, control);

  EXPECT_GT(next.covariance.data(0, 0), 1.0);
  EXPECT_GT(next.covariance.data(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(next.covariance.data(0, 1), dt);
  EXPECT_DOUBLE_EQ(next.covariance.data(1, 0), next.covariance.data(0, 1));
}

TEST(EKF, ProcessNoise) {
  double dt = 0.05;

  double x = -1.0;
  double y = 2.0;
  {
    State state({x, y});
    Covariance covariance;
    Control control;

    ExtendedKalmanFilter::CovarianceT process_noise =
        ExtendedKalmanFilter::ProcessModel::covariance(dt, {state, covariance},
                                                       control);

    EXPECT_EQ(process_noise.rows(), 1);
    EXPECT_EQ(process_noise.cols(), 1);
    EXPECT_DOUBLE_EQ(process_noise(0, 0), 0.25);
  }
}

TEST(EKF, ControlJacobian) {
  double dt = 0.05;

  double x = -1.0;
  double y = 2.0;
  {
    State state({x, y});
    Covariance covariance;
    Control control;

    ExtendedKalmanFilter::ControlJacobianT jacobian =
        ExtendedKalmanFilter::ProcessModel::control_jacobian(
            dt, {state, covariance}, control);
    EXPECT_EQ(jacobian.rows(), 2);
    EXPECT_EQ(jacobian.cols(), 1);

    EXPECT_DOUBLE_EQ(jacobian(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(jacobian(1, 0), dt);
  }
}

}  // namespace unit
