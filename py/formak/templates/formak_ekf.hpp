#pragma once

#include <Eigen/Core>  // Matrix
#include <cstddef>     // size_t

namespace formak {

struct State {
  // clang-format off
  {{State_members}}
  // clang-format on
};

struct Covariance
    : public Eigen::Matrix<double, {{State_size}}, {{State_size}}> {
  // clang-format off
  {{Covariance_members}}
  // clang-format on
};

struct Control {
  // clang-format off
  {{Control_members}}
  // clang-format on
};

struct StateAndVariance {
  State state;
  Covariance covariance;
};

enum class SensorId {
  // clang-format off
    {{SensorId_members}}
  // clang-format on
};

template <SensorId Identifier, typename ReadingT>
struct SensorReading {
  SensorId id = Identifier;
  ReadingT reading;
};

// clang-format off
{% for reading_type in reading_types %}
struct {{reading_type.typename}}SensorModel;

struct {{reading_type.typename}} {
  using SensorModel = {{reading_type.typename}}SensorModel;
  using CovarianceT = Eigen::Matrix<double, {{reading_type.size}}, {{reading_type.size}}>;
  using SensorJacobianT = Eigen::Matrix<double, {{reading_type.size}}, {{State_size}}>;
  using KalmanGainT = Eigen::Matrix<double, {{State_size}}, {{reading_type.size}}>;
  using InnovationT = Eigen::Matrix<double, {{reading_type.size}}, 1>;
  constexpr static size_t size = {{reading_type.size}};

  {{reading_type.members}}
};

struct {{reading_type.typename}}SensorModel {
    static {{reading_type.typename}} model(
      const StateAndVariance& input,
      const SensorReading<{{reading_type.identifier}}, {{reading_type.typename}}>& input_reading);

    static typename {{reading_type.typename}}::SensorJacobianT jacobian(
            const StateAndVariance& input,
            const SensorReading<{{reading_type.identifier}}, {{reading_type.typename}}>& input_reading);

    static typename {{reading_type.typename}}::CovarianceT covariance(
            const StateAndVariance& input,
            const SensorReading<{{reading_type.identifier}}, {{reading_type.typename}}>& input_reading);
};

{% endfor %}
// clang-format on

class ExtendedKalmanFilter {
 public:
  StateAndVariance process_model(double dt, const StateAndVariance& input,
                                 const Control& input_control);

  template <SensorId Identifier, typename ReadingT>
  StateAndVariance sensor_model(
      const StateAndVariance& input,
      const SensorReading<Identifier, ReadingT>& input_reading) {
    const State& state = input.state;                 // mu
    const Covariance& covariance = input.covariance;  // Sigma
    const ReadingT& reading = input_reading.reading;  // z

    // z_est = sensor_model()
    const ReadingT reading_est =
        ReadingT::SensorModel::model(input, input_reading);  // z_est

    // H = Jacobian(z_est w.r.t. state)
    const typename ReadingT::SensorJacobianT H =
        ReadingT::SensorModel::jacobian(input, input_reading);

    // Project State Noise into Sensor Space
    // S = H * Sigma * H.T + Q_t
    const typename ReadingT::CovarianceT sensor_estimate_covariance =
        H * covariance * H.transpose() +
        ReadingT::SensorModel::covariance(input, input_reading);

    // S_inv = inverse(S)
    const typename ReadingT::CovarianceT S_inv =
        sensor_estimate_covariance.inverse();

    // Kalman Gain
    // K = Sigma * H.T * S_inv
    const typename ReadingT::KalmanGainT kalman_gain =
        covariance * H.transpose() * S_inv;

    // Innovation
    // innovation = z - z_est
    const typename ReadingT::InnovationT innovation = reading - reading_est;

    // Update State Estimate
    // next_state = state + K * innovation
    const State next_state = state + kalman_gain * innovation;

    // Update Covariance
    // next_covariance = Sigma - K * H * Sigma
    const Covariance next_covariance =
        covariance - kalman_gain * H * covariance;

    // TODO(buck): Measurement Likelihood (optional)

    // Here be the StateAndVariance math
    return StateAndVariance{.state = next_state, .covariance = next_covariance};
  }
};

}  // namespace formak
