#include <gtest/gtest.h>
#include <unit/simple-ekf.h>  // Generated

namespace unit {

TEST(EKF, State_default_initialization_is_zero) {
  State state;
  EXPECT_EQ(state.x, 0.0);
  EXPECT_EQ(state.y, 0.0);
}

TEST(EKF, Covariance_default_initialization_is_identity) {
  Covariance covariance;
  EXPECT_EQ(covariance.x, 1.0);
  EXPECT_EQ(covariance.y, 1.0);

  EXPECT_EQ(covariance.data(1, 0), 0.0);
  EXPECT_EQ(covariance.data(0, 1), 0.0);
}

TEST(EKF, process_with_control) {
  ExtendedKalmanFilter ekf;
  double dt = 0.1;

  Control control{0.2};

  Covariance covariance;
  {
    State state({0.0, 0.0});

    auto next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x, 0.0);
    EXPECT_DOUBLE_EQ(next.state.y, 0.02);
  }

  {
    State state({0.0, 1.0});

    auto next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x, 0.0);
    EXPECT_DOUBLE_EQ(next.state.y, 1.02);
  }

  {
    State state({1.0, 0.0});

    auto next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x, 0.0);
    EXPECT_DOUBLE_EQ(next.state.y, 0.02);
  }

  {
    State state({1.0, 1.0});

    auto next = ekf.process_model(dt, {state, covariance}, control);

    EXPECT_DOUBLE_EQ(next.state.x, 1.0);
    EXPECT_DOUBLE_EQ(next.state.y, 1.02);
  }
}

}  // namespace unit
