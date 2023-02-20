#include <formak/cpp-ekf.h>  // Generated
#include <gtest/gtest.h>

namespace featuretest {

TEST(CppModel, SimpleEKF) {
  formak::State state({.v = 1.0});
  formak::Covariance state_variance;
  formak::Control control;

  formak::ExtendedKalmanFilter ekf;

  ASSERT_DOUBLE_EQ(state.z, 0.0);

  auto state_and_variance = ekf.process_model(
      0.1, {.state = state, .covariance = state_variance}, control);
  formak::State next_state = state_and_variance.state;
  formak::Covariance next_variance = state_and_variance.covariance;

  EXPECT_GT(next_state.z, 0.0);

  double previous_z = next_state.z;

  formak::SensorReading<formak::SensorId::SIMPLE, formak::Simple>
      zero_sensor_reading;
  auto next_state_and_variance = ekf.sensor_model(
      {.state = next_state, .covariance = next_variance}, zero_sensor_reading);

  EXPECT_GE(next_state_and_variance.state.z, 0.0);
  EXPECT_LT(next_state_and_variance.state.z, previous_z);
}

}  // namespace featuretest
