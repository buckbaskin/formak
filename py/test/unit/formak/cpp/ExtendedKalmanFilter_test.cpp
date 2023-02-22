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

TEST(EKF, Control_constructor) {
  {
    Control control;
    EXPECT_DOUBLE_EQ(control.a(), 0.0);
  }

  {
    Control control({0.2});
    EXPECT_DOUBLE_EQ(control.a(), 0.2);
  }
}

TEST(EKF, State_constructor) {
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
    EXPECT_DOUBLE_EQ(state.x(), 1.0);
    EXPECT_DOUBLE_EQ(state.y(), 0.0);

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
  ExtendedKalmanFilter ekf;

  Covariance covariance;
  State state({0.0, 0.0});

  double reading = 1.0;

  SensorReading<(SensorId::SIMPLE), Simple> simple_reading{Simple({reading})};
  auto next = ekf.sensor_model({state, covariance}, simple_reading);

  EXPECT_LT(abs(reading - next.state.data(0, 0)),
            abs(reading - state.data(0, 0)));

  // TODO(buck): Check what this should be
  SensorReading<(SensorId::COMBINED), Combined> combined{Combined({reading})};
  next = ekf.sensor_model({state, covariance}, combined);

  EXPECT_LT(abs(reading - next.state.data(0, 0)),
            abs(reading - state.data(0, 0)));
  EXPECT_LT(abs(reading - next.state.data(1, 0)),
            abs(reading - state.data(1, 0)));
}

TEST(EKF, sensor_model_detail) {
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

TEST(EKF, sensor_covariances) {
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

TEST(EKF, sensor_jacobian) {
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
    EXPECT_DOUBLE_EQ(out(0, 1), 0.0);
    FAIL();
  }
}

TEST(EKF, process_jacobian) {
  ExtendedKalmanFilter ekf;

  double dt = 0.05;

  double x = -1.0;
  double y = 2.0;
  {
    State state({x, y});
    Covariance covariance;
    Control control;
    EXPECT_DOUBLE_EQ(control.a(), 0.0);

    StateAndVariance next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x(), -3.0);
  }
}

}  // namespace unit
