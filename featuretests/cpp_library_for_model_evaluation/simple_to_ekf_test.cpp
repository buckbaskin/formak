#include <formak/cpp-ekf.h>  // Generated
#include <gtest/gtest.h>

namespace featuretest {

TEST(CppModel, SimpleEKF) {
  formak::State state;
  formak::Covariance state_variance;
  formak::Control control;

  formak::ExtendedKalmanFilter ekf;

  auto state_and_variance = ekf.process_model(
      0.1, {.state = state, .covariance = state_variance}, control);
  formak::State next_state = state_and_variance.state;
  formak::Covariance next_variance = state_and_variance.covariance;

  formak::SensorReading<formak::SensorId::SIMPLE> sensor_reading;
  auto next_state_and_variance = ekf.sensor_model(
      {.state = next_state, .covariance = next_variance}, sensor_reading);
}

}  // namespace featuretest
