#include <gtest/gtest.h>
#include <unit/simple-ekf.h>  // Generated

namespace unit {

TEST(EKF, State_default_initialization_is_zero) {
  State state;
  EXPECT_EQ(state.x(), 0.0);
  EXPECT_EQ(state.y(), 0.0);
}

TEST(EKF, Covariance_default_initialization_is_identity) {
  Covariance covariance;
  EXPECT_EQ(covariance.x(), 1.0);
  EXPECT_EQ(covariance.y(), 1.0);

  EXPECT_EQ(covariance.data(1, 0), 0.0);
  EXPECT_EQ(covariance.data(0, 1), 0.0);
}

TEST(EKF, process_with_control) {
  ExtendedKalmanFilter ekf;
  double dt = 0.1;

  Control control({0.2});

  Covariance covariance;
  {
    State state({0.0, 0.0});

    auto next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x(), 0.0);
    EXPECT_DOUBLE_EQ(next.state.y(), 0.02);
  }

  {
    State state({0.0, 1.0});

    auto next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x(), 0.0);
    EXPECT_DOUBLE_EQ(next.state.y(), 1.02);
  }

  {
    State state({1.0, 0.0});

    auto next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x(), 0.0);
    EXPECT_DOUBLE_EQ(next.state.y(), 0.02);
  }

  {
    State state({1.0, 1.0});

    auto next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x(), 1.0);
    EXPECT_DOUBLE_EQ(next.state.y(), 1.02);
  }
}

TEST(EKF, sensor) {
  //     ekf = python.ExtendedKalmanFilter(
  //         state_model=ui.Model(
  //             ui.Symbol("dt"),
  //             set(ui.symbols(["x", "y"])),
  //             set(ui.symbols(["a"])),
  //             {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
  //         ),
  //         process_noise=np.eye(1),
  //         sensor_models={
  //             "simple": {"reading1": ui.Symbol("x")},
  //             "combined": {"reading2": ui.Symbol("x") + ui.Symbol("y")},
  //         },
  //         sensor_noises={"simple": np.eye(1), "combined": np.eye(1)},
  //         config=config,
  //     )
  ExtendedKalmanFilter ekf;

  Covariance covariance;
  State state({0.0, 0.0});

  double reading = 1.0;

  SensorReading<(SensorId::SIMPLE), Simple> simple_reading{Simple{reading}};
  auto next = ekf.sensor_model({state, covariance}, simple_reading);

  EXPECT_LT(abs(reading - next.state.data(0, 0)),
            abs(reading - state.data(0, 0)));

  // TODO(buck): Check what this should be
  SensorReading<(SensorId::COMBINED), Combined> combined{Combined{reading}};
  next = ekf.sensor_model({state, covariance}, combined);

  EXPECT_LT(abs(reading - next.state.data(0, 0)),
            abs(reading - state.data(0, 0)));
  EXPECT_LT(abs(reading - next.state.data(1, 0)),
            abs(reading - state.data(1, 0)));
}

TEST(EKF, sensor_model_detail) {
  //     ekf = python.ExtendedKalmanFilter(
  //         state_model=ui.Model(
  //             ui.Symbol("dt"),
  //             set(ui.symbols(["x", "y"])),
  //             set(ui.symbols(["a"])),
  //             {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
  //         ),
  //         process_noise=np.eye(1),
  //         sensor_models={
  //             "simple": {"reading1": ui.Symbol("x")},
  //             "combined": {"reading2": ui.Symbol("x") + ui.Symbol("y")},
  //         },
  //         sensor_noises={"simple": np.eye(1), "combined": np.eye(1)},
  //         config=config,
  //     )
  ExtendedKalmanFilter ekf;

  double x = -1.0;
  double y = 2.0;
  State state({x, y});

  {
    SensorReading<(SensorId::SIMPLE), Simple> simple_reading{Simple{}};
    Simple predicted = Simple::SensorModel::model(state, simple_reading);
    EXPECT_DOUBLE_EQ(predicted.reading1(), x);
  }

  {
    SensorReading<(SensorId::COMBINED), Combined> combined_reading{Combined{}};
    Combined predicted = Combined::SensorModel::model(state, combined_reading);
    EXPECT_DOUBLE_EQ(predicted.reading2(), x + y);
  }
}
// def test_EKF_process_jacobian():
//     config = python.Config()
//     dt = 0.1
//
//     ekf = python.ExtendedKalmanFilter(
//         state_model=ui.Model(
//             ui.Symbol("dt"),
//             set(ui.symbols(["x", "y"])),
//             set(ui.symbols(["a"])),

}  // namespace unit
