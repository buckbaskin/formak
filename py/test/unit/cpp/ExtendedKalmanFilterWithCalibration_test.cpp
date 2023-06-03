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
}

INSTANTIATE_TEST_SUITE_P(StateTestCases, ProcessWithCalibrationTest,
                         ::testing::Values(Options{.xStart = 0.0, .xEnd = 5.5},
                                           Options{.xStart = 3.1, .xEnd = 8.6},
                                           Options{.xStart = -1.0,
                                                   .xEnd = 4.5}));
}  // namespace process_with_calibration_test

// ui_model = ui.Model(
//     dt=dt,
//     state=set([x]),
//     control=set(),
//     calibration=set([a, b]),
//     state_model={x: x + a + b},
// )
//
// cpp_implementation = cpp.compile_ekf(
//     state_model=ui_model,
//     process_noise={},
//     sensor_models={"y": {y: x + b}},
//     sensor_noises={"y": np.eye(1)},
//     calibration_map={ui.Symbol("a"): 5.0, ui.Symbol("b"): 0.5},
//     config={},
// )

namespace ekf_sensor_test {
TEST(EKF, SensorY) {
  ExtendedKalmanFilter ekf;

  State state({0.0});
  Covariance covariance;

  Calibration calibration({.a = 5.0, .b = 0.5});

  double reading = 1.0;

  Y simple_reading({reading});
  EXPECT_EQ(simple_reading.data.rows(), 1);
  EXPECT_EQ(simple_reading.data.cols(), 1);

  auto next =
      ekf.sensor_model({state, covariance}, calibration, simple_reading);

  EXPECT_LT(abs(reading - next.state.data(0, 0)),
            abs(reading - state.data(0, 0)));
}
}  // namespace ekf_sensor_test

TEST(EKF, SensorModelDetail) {
  constexpr double X = -1.0;
  State state({X});
  Covariance covariance;

  constexpr double B = 0.5;
  Calibration calibration({.a = 5.0, .b = B});

  {
    Y simple_reading{};
    Y predicted =
        Y::SensorModel::model({state, covariance}, calibration, simple_reading);
    EXPECT_DOUBLE_EQ(predicted.y(), X + B);
  }
}

TEST(EKF, SensorCovariances) {
  double x = -1.0;
  State state({x});
  Covariance covariance;

  Calibration calibration({.a = 5.0, .b = 0.5});

  {
    Y simple_reading{};
    Y::CovarianceT out = Y::SensorModel::covariance(
        {state, covariance}, calibration, simple_reading);
    EXPECT_DOUBLE_EQ(out(0, 0), 1.0);
  }
}

TEST(EKF, SensorJacobian) {
  double x = -1.0;
  State state({x});
  Covariance covariance;

  Calibration calibration({.a = 5.0, .b = 0.5});

  {
    Y simple_reading{};
    Y::SensorJacobianT out = Y::SensorModel::jacobian(
        {state, covariance}, calibration, simple_reading);
    EXPECT_DOUBLE_EQ(out(0, 0), 1.0);
  }
}

TEST(EKF, ProcessJacobian) {
  ExtendedKalmanFilter ekf;

  double dt = 0.05;

  double x = -1.0;

  State state({x});
  // Assumes default initialization of covariance (identity) and calibration (0)
  Covariance covariance;
  Calibration calibration;

  StateAndVariance next =
      ekf.process_model(dt, {state, covariance}, calibration);

  EXPECT_DOUBLE_EQ(next.covariance.data(0, 0), 1.0);
}

}  // namespace unit
