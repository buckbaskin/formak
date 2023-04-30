#pragma once

#include <Eigen/Dense>    // Matrix
#include <any>            // any
#include <cstddef>        // size_t
#include <iostream>       // std::cout, debugging
#include <optional>       // optional
#include <unordered_map>  // unordered_map

// clang-format off
namespace {{namespace}} {
// clang-format-on
  struct StateOptions {
    // clang-format off
    {{ StateOptions_members }}
    // clang-format on
  };

  struct State {
    static constexpr size_t rows = {{State_size}};
    static constexpr size_t cols = 1;
    using DataT = Eigen::Matrix<double, rows, cols>;

    State();
    State(const StateOptions& options);
    // clang-format off
    {{State_members}}
    // clang-format on
    DataT data = DataT::Zero();
  };

  struct Covariance {
    using DataT = Eigen::Matrix<double, {{State_size}}, {{State_size}}>;

    // clang-format off
    {{Covariance_members}}
    // clang-format on
    DataT data = DataT::Identity();
  };

  // clang-format off
{% if enable_control %}
  // clang-format on
  struct ControlOptions {
    // clang-format off
    {{ ControlOptions_members }}
    // clang-format on
  };

  struct Control {
    Control();
    Control(const ControlOptions& options);
    // clang-format off
    {{Control_members}}
    // clang-format on
    Eigen::Matrix<double, {{Control_size}}, 1> data =
        Eigen::Matrix<double, {{Control_size}}, 1>::Zero();
  };
  // clang-format off
{% endif %}  // clang-format on

  // clang-format off
{% if enable_calibration %}
  // clang-format on
  struct CalibrationOptions {
    // clang-format off
    {{ CalibrationOptions_members }}
    // clang-format on
  };

  struct Calibration {
    Calibration();
    Calibration(const CalibrationOptions& options);
    // clang-format off
    {{Calibration_members}}
    // clang-format on
    Eigen::Matrix<double, {{Calibration_size}}, 1> data =
        Eigen::Matrix<double, {{Calibration_size}}, 1>::Zero();
  };
  // clang-format off
{% endif %}
  // clang-format on
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
    ReadingT reading;
    static constexpr SensorId id = Identifier;
  };

  // clang-format off
{% for reading_type in reading_types %}
  // ReadingTSensorModel
  struct {{reading_type.typename}}SensorModel;

  struct {{reading_type.typename}}Options {
    {{ reading_type.Options_members }}
  };

  // ReadingT
  struct {{reading_type.typename}} {
    using DataT = Eigen::Matrix<double, {{reading_type.size}}, 1>;
    using CovarianceT = Eigen::Matrix<double, {{reading_type.size}}, {{reading_type.size}}>;
    using InnovationT = Eigen::Matrix<double, {{reading_type.size}}, 1>;
    using KalmanGainT = Eigen::Matrix<double, {{State_size}}, {{reading_type.size}}>;
    using SensorJacobianT = Eigen::Matrix<double, {{reading_type.size}}, {{State_size}}>;
    using SensorModel = {{reading_type.typename}}SensorModel;

    {{reading_type.typename}}();
    {{reading_type.typename}}(const {{reading_type.typename}}Options& options);

    {{reading_type.members}}

    DataT data = DataT::Zero();

    constexpr static size_t size = {{reading_type.size}};
  };

  std::ostream& operator<<(std::ostream& o, const {{reading_type.typename}}& reading) {
      o << "Reading(data[{{reading_type.size}}, 1] = " << reading.data << ")";
      return o;
  }

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
  class ExtendedKalmanFilterProcessModel;

  class ExtendedKalmanFilter {
   public:
    using CovarianceT =
        Eigen::Matrix<double, {{Control_size}}, {{Control_size}}>;
    using ProcessJacobianT =
        Eigen::Matrix<double, {{State_size}}, {{State_size}}>;
    using ControlJacobianT =
        Eigen::Matrix<double, {{State_size}}, {{Control_size}}>;
    using ProcessModel = ExtendedKalmanFilterProcessModel;

    StateAndVariance process_model(
        double dt,
        const StateAndVariance& input
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        const Calibration& input_calibration
        // clang-format off
{% endif %}  // clang-format on
                     // clang-format off
{% if enable_control %}
                     // clang-format on
        ,
        const Control& input_control
        // clang-format off
{% endif %}  // clang-format on
    );

    template <SensorId Identifier, typename ReadingT>
    StateAndVariance sensor_model(
        const StateAndVariance& input,
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        const Calibration& input_calibration,
        // clang-format off
{% endif %}  // clang-format on
        const SensorReading<Identifier, ReadingT>& input_reading) {
      const State& state = input.state;                 // mu
      const Covariance& covariance = input.covariance;  // Sigma
      const ReadingT& reading = input_reading.reading;  // z

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
          reading.data - reading_est.data;
      _innovations[Identifier] = innovation;

      // Update State Estimate
      // next_state = state + K * innovation
      State next_state;
      next_state.data = state.data + kalman_gain * innovation;

      // Update Covariance
      // next_covariance = Sigma - K * H * Sigma
      Covariance next_covariance;
      next_covariance.data =
          covariance.data - kalman_gain * H * covariance.data;

      // TODO(buck): Measurement Likelihood (optional)

      // Here be the StateAndVariance math
      return StateAndVariance{.state = next_state,
                              .covariance = next_covariance};
    }

    template <SensorId Identifier, typename ReadingT>
    std::optional<typename ReadingT::InnovationT> innovations() {
      if (_innovations.count(Identifier) > 0) {
        return std::any_cast<typename ReadingT::InnovationT>(
            _innovations[Identifier]);
      }
      return {};
    }

   private:
    std::unordered_map<SensorId, std::any> _innovations;
  };

  class ExtendedKalmanFilterProcessModel {
   public:
    static State model(
        double dt,
        const StateAndVariance& input
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        const Calibration& input_calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        const Control& input_control
        // clang-format off
{% endif %}  // clang-format on
    );

    static typename ExtendedKalmanFilter::ProcessJacobianT process_jacobian(
        double dt,
        const StateAndVariance& input
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        const Calibration& input_calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        const Control& input_control
        // clang-format off
{% endif %}  // clang-format on
    );

    static typename ExtendedKalmanFilter::ControlJacobianT control_jacobian(
        double dt,
        const StateAndVariance& input
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        const Calibration& input_calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        const Control& input_control
        // clang-format off
{% endif %}  // clang-format on
    );

    static typename ExtendedKalmanFilter::CovarianceT covariance(
        double dt,
        const StateAndVariance& input
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        const Calibration& input_calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        const Control& input_control
        // clang-format off
{% endif %}  // clang-format on
    );
  };

}  // namespace {{namespace}}
