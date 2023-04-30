#include <gtest/gtest.h>
#include <unit/calibration-ekf.h>  // Generated

namespace unit {

namespace process_with_calibration_test {
struct Options {
  double xStart;

  double xEnd;
};

class ProcessWithCalibrationTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<Options> {};

TEST_P(ProcessWithCalibrationTest, Test) {
  ExtendedKalmanFilter ekf;
  double dt = 0.1;

  Calibration calibration({.a = 5.0, .b = 0.5});

  Covariance covariance;

  State state({.x = GetParam().xStart});

  auto next = ekf.process_model(dt, {state, covariance}, calibration);

  EXPECT_DOUBLE_EQ(next.state.x(), GetParam().xEnd);
  EXPECT_DOUBLE_EQ(next.state.y(), GetParam().yEnd);
}

INSTANTIATE_TEST_SUITE_P(StateTestCases, ProcessWithCalibrationTest,
                         ::testing::Values(Options{0.0, 0.0, 0.0, 0.02},
                                           Options{0.0, 1.0, 0.1, 1.02},
                                           Options{1.0, 0.0, 1.0, 0.02},
                                           Options{1.0, 1.0, 1.1, 1.02}));
}  // namespace process_with_calibration_test

namespace ekf_sensor_test {
TEST(EKF, SensorSimple) {
  ExtendedKalmanFilter ekf;

  Covariance covariance;
  State state({0.0, 0.0});

  double reading = 1.0;

  SensorReading<(SensorId::SIMPLE), Simple> simple_reading{Simple({reading})};
  EXPECT_EQ(simple_reading.reading.data.rows(), 1);
  EXPECT_EQ(simple_reading.reading.data.cols(), 1);

  auto next = ekf.sensor_model({state, covariance}, simple_reading);

  EXPECT_LT(abs(reading - next.state.data(0, 0)),
            abs(reading - state.data(0, 0)));
}
TEST(EKF, SensorCombined) {
  ExtendedKalmanFilter ekf;

  Covariance covariance;
  State state({0.0, 0.0});

  double reading = 1.0;

  SensorReading<(SensorId::COMBINED), Combined> combined{Combined({reading})};
  EXPECT_EQ(combined.reading.data.rows(), 1);
  EXPECT_EQ(combined.reading.data.cols(), 1);

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
    SensorReading<(SensorId::SIMPLE), Simple> simple_reading{Simple{}};
    Simple predicted =
        Simple::SensorModel::model({state, covariance}, simple_reading);
    EXPECT_DOUBLE_EQ(predicted.reading1(), x);
  }

  {
    SensorReading<(SensorId::COMBINED), Combined> combined_reading{Combined{}};
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
    SensorReading<(SensorId::SIMPLE), Simple> simple_reading{Simple{}};
    Simple::CovarianceT out =
        Simple::SensorModel::covariance({state, covariance}, simple_reading);
    EXPECT_DOUBLE_EQ(out(0, 0), 1.0);
  }

  {
    SensorReading<(SensorId::COMBINED), Combined> combined_reading{Combined{}};
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
    SensorReading<(SensorId::SIMPLE), Simple> simple_reading{Simple{}};
    Simple::SensorJacobianT out =
        Simple::SensorModel::jacobian({state, covariance}, simple_reading);
    EXPECT_DOUBLE_EQ(out(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(out(0, 1), 0.0);
  }

  {
    SensorReading<(SensorId::COMBINED), Combined> combined_reading{Combined{}};
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
  // Assumes default initialization of covariance (identity) and calibration (0)
  Covariance covariance;
  Calibration calibration;

  StateAndVariance next =
      ekf.process_model(dt, {state, covariance}, calibration);

  EXPECT_GT(next.covariance.data(0, 0), 1.0);
  EXPECT_GT(next.covariance.data(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(next.covariance.data(0, 1), dt);
  EXPECT_DOUBLE_EQ(next.covariance.data(1, 0), next.covariance.data(0, 1));
}

}  // namespace unit
