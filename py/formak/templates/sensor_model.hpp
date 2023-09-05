const State& mu = state.state;
const Covariance& Sigma = state.covariance;

// z_est = sensor_model()
const ReadingT reading_est =
    ReadingT::SensorModel::model(state,
                                 // clang-format off
{% if enable_calibration %}
                                 // clang-format on
                                 calibration,
                                 // clang-format off
{% endif %}  // clang-format on
                                 reading);      // z_est

// H = Jacobian(z_est w.r.t. state)
const typename ReadingT::SensorJacobianT H =
    ReadingT::SensorModel::jacobian(state,
                                    // clang-format off
{% if enable_calibration %}
                                    // clang-format on
                                    calibration,
                                    // clang-format off
{% endif %}  // clang-format on
                                    reading);

// Project State Noise into Sensor Space
// S = H * Sigma * H.T + Q_t
const typename ReadingT::CovarianceT sensor_estimate_covariance =
    H * Sigma.data * H.transpose() +
    ReadingT::SensorModel::covariance(state,
                                      // clang-format off
{% if enable_calibration %}
                                      // clang-format on
                                      calibration,
                                      // clang-format off
{% endif %}  // clang-format on
                                      reading);

// S_inv = inverse(S)
const typename ReadingT::CovarianceT S_inv =
    sensor_estimate_covariance.inverse();

// Kalman Gain
// K = Sigma * H.T * S_inv
const typename ReadingT::KalmanGainT kalman_gain =
    Sigma.data * H.transpose() * S_inv;

// Innovation
// innovation = z - z_est
const typename ReadingT::InnovationT innovation =
    reading.data - reading_est.data;
_innovations[ReadingT::Identifier] = innovation;

if constexpr (cpp::Config::innovation_filtering > 0.0) {
  if (innovation_filtering::edit::removeInnovation(
          cpp::Config::innovation_filtering, ReadingT::size, innovation,
          sensor_estimate_covariance)) {
    // Skip update
    return state;
  }
}

// Update State Estimate
// next_state = state + K * innovation
State next_state;
next_state.data = mu.data + kalman_gain * innovation;

// Update Covariance
// next_covariance = Sigma - K * H * Sigma
Covariance next_covariance;
next_covariance.data = Sigma.data - kalman_gain * H * Sigma.data;

// TODO(buck): Measurement Likelihood (optional)

// Here be the StateAndVariance math
return StateAndVariance{.state = next_state, .covariance = next_covariance};
