const State& state = input.state;                 // mu
const Covariance& covariance = input.covariance;  // Sigma

// z_est = sensor_model()
const ReadingT reading_est =
    ReadingT::SensorModel::model(input,
                                 // clang-format off
{% if enable_calibration %}
                                 // clang-format on
                                 input_calibration,
                                 // clang-format off
{% endif %}      // clang-format on
                                 input_reading);  // z_est

// H = Jacobian(z_est w.r.t. state)
const typename ReadingT::SensorJacobianT H =
    ReadingT::SensorModel::jacobian(input,
                                    // clang-format off
{% if enable_calibration %}
                                    // clang-format on
                                    input_calibration,
                                    // clang-format off
{% endif %}  // clang-format on
                                    input_reading);

// Project State Noise into Sensor Space
// S = H * Sigma * H.T + Q_t
const typename ReadingT::CovarianceT sensor_estimate_covariance =
    H * covariance.data * H.transpose() +
    ReadingT::SensorModel::covariance(input,
                                      // clang-format off
{% if enable_calibration %}
                                      // clang-format on
                                      input_calibration,
                                      // clang-format off
{% endif %}  // clang-format on
                                      input_reading);

// S_inv = inverse(S)
const typename ReadingT::CovarianceT S_inv =
    sensor_estimate_covariance.inverse();

// Kalman Gain
// K = Sigma * H.T * S_inv
const typename ReadingT::KalmanGainT kalman_gain =
    covariance.data * H.transpose() * S_inv;

// Innovation
// innovation = z - z_est
const typename ReadingT::InnovationT innovation =
    input_reading.data - reading_est.data;
_innovations[ReadingT::Identifier] = innovation;

// Update State Estimate
// next_state = state + K * innovation
State next_state;
next_state.data = state.data + kalman_gain * innovation;

// Update Covariance
// next_covariance = Sigma - K * H * Sigma
Covariance next_covariance;
next_covariance.data = covariance.data - kalman_gain * H * covariance.data;

// TODO(buck): Measurement Likelihood (optional)

// Here be the StateAndVariance math
return StateAndVariance{.state = next_state, .covariance = next_covariance};
