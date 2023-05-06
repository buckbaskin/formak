#include <featuretest/cpp-rocket-model.h>  // Generated
#include <gtest/gtest.h>

namespace featuretest {

TEST(Featuretest, RocketModel) {
  double starting_velocity = 1.0;

  featuretest::State state({.CON_vel_z = starting_velocity});
  featuretest::Covariance state_variance;
  featuretest::Calibration calibration({
      // orientation would need to invert a rotation matrix
      .IMU_ori_pitch = 0.0,
      .IMU_ori_roll = 0.0,
      .IMU_ori_yaw = 0.0,
      // pos_IMU_from_CON_in_CON     [m]     [-0.08035, 0.28390, -1.42333 ]
      .IMU_pos_x = -0.08035,
      .IMU_pos_y = 0.28390,
      .IMU_pos_z = -1.42333,
  });
  featuretest::Control control;

  featuretest::ExtendedKalmanFilter ekf;

  ASSERT_DOUBLE_EQ(state.CON_pos_pos_z(), 0.0);

  auto state_and_variance =
      ekf.process_model(0.1, {.state = state, .covariance = state_variance},
                        calibration, control);
  featuretest::State next_state = state_and_variance.state;
  featuretest::Covariance next_variance = state_and_variance.covariance;

  EXPECT_GT(next_state.CON_pos_pos_z(), 0.0);

  featuretest::Altitude zero_sensor_reading;
  auto next_state_and_variance =
      ekf.sensor_model({.state = next_state, .covariance = next_variance},
                       calibration, zero_sensor_reading);

  EXPECT_GE(next_state_and_variance.state.CON_vel_z(), 0.0);
  EXPECT_LT(next_state_and_variance.state.CON_vel_z(), starting_velocity);
}

}  // namespace featuretest
