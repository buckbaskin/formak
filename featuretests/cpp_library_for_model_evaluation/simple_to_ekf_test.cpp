#include <formak/cpp-model.h>  // Generated
#include <gtest/gtest.h>

namespace featuretest {

TEST(CppModel, Simple) {
  formak::State state;
  formak::StateVariance state_variance;
  formak::Control control;

  formak::EKF ekf;

  auto state_and_variance = ekf.process_model(
      0.1, {.state = state, .covariance = state_variance}, control);
  formak::State next_state = state_and_variance.state;
  formak::State next_variance = state_and_variance.variance;

  // TODO(buck): Define [enum] for selecting sensor model
  // TODO(buck): Define sensor reading type(s)
  auto next_state_and_variance =
      ekf.sensor_model({.state = next_state, .covariance = next_variance},
                       ekf.SIMPLE, sensor_reading);
}

}  // namespace featuretest
