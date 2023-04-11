#include <{{header_include}}>

// clang-format off
namespace {{namespace}} {
// clang-format-on
  State::State() : data(Eigen::Matrix<double, {{State_size}}, 1>::Zero()) {
  }
  State::State(const StateOptions& options)
      : {{State_options_constructor_initializer_list}} {
  }

  Control::Control() : data(Eigen::Matrix<double, {{Control_size}}, 1>::Zero()) {
  }
  Control::Control(const ControlOptions& options)
      : {{Control_options_constructor_initializer_list}} {
  }

  StateAndVariance ExtendedKalmanFilter::process_model(
      double dt, const StateAndVariance& input, const Control& input_control){
    const Covariance& covariance = input.covariance;
    // G = process_jacobian
    ExtendedKalmanFilter::ProcessJacobianT G = ExtendedKalmanFilter::ProcessModel::process_jacobian(dt, input, input_control);
    // V = control_jacobian
    ExtendedKalmanFilter::ControlJacobianT V = ExtendedKalmanFilter::ProcessModel::control_jacobian(dt, input, input_control);
    // M = process_noise
    ExtendedKalmanFilter::CovarianceT M = ExtendedKalmanFilter::ProcessModel::covariance(dt, input, input_control);

    // Update State Estimate
    // next_state = process_model(input, input_control)
    State next_state = ExtendedKalmanFilter::ProcessModel::model(dt, input, input_control);

    // Update Covariance
    // Sigma = G * Sigma * G.T + V * M * V.T
    Covariance next_covariance;
    next_covariance.data =
        G * covariance.data * G.transpose() + V * M * V.transpose();

    return {.state = next_state, .covariance = next_covariance};
  }

  State ExtendedKalmanFilterProcessModel::model(double dt,const StateAndVariance& input,
                       const Control& input_control) {
    // clang-format off
{{ExtendedKalmanFilterProcessModel_model_body}}
    // clang-format on
  }

  typename ExtendedKalmanFilter::ProcessJacobianT
  ExtendedKalmanFilterProcessModel::process_jacobian(
      double dt, const StateAndVariance& input, const Control& input_control) {
    // clang-format off
{{ExtendedKalmanFilterProcessModel_process_jacobian_body}}
    // clang-format on
  }

  typename ExtendedKalmanFilter::ControlJacobianT
  ExtendedKalmanFilterProcessModel::control_jacobian(
      double dt, const StateAndVariance& input, const Control& input_control) {
    // clang-format off
{{ExtendedKalmanFilterProcessModel_control_jacobian_body}}
    // clang-format on
  }

  typename ExtendedKalmanFilter::CovarianceT
  ExtendedKalmanFilterProcessModel::covariance(
      double dt, const StateAndVariance& input, const Control& input_control){
      // clang-format off
{{ExtendedKalmanFilterProcessModel_covariance_body}}
      // clang-format on
  }

  // clang-format off
{% for reading_type in reading_types %}
  {{reading_type.typename}}::{{reading_type.typename}}() : data(Eigen::Matrix<double, {{reading_type.size}}, 1>::Zero()) {
  }
  {{reading_type.typename}}::{{reading_type.typename}}(const {{reading_type.typename}}Options& options) : {{reading_type.initializer_list}} {
  }

  {{reading_type.typename}} {{reading_type.typename}}SensorModel::model(
        const StateAndVariance& input,
        const SensorReading<{{reading_type.identifier}}, {{reading_type.typename}}>& input_reading) {
{{reading_type.SensorModel_model_body}}
  }

  {{reading_type.typename}}::CovarianceT {{reading_type.typename}}SensorModel::covariance(
        const StateAndVariance& input,
        const SensorReading<{{reading_type.identifier}}, {{reading_type.typename}}>& input_reading) {
{{reading_type.SensorModel_covariance_body}}
  }

  {{reading_type.typename}}::SensorJacobianT {{reading_type.typename}}SensorModel::jacobian(
        const StateAndVariance& input,
        const SensorReading<{{reading_type.identifier}}, {{reading_type.typename}}>& input_reading) {
{{reading_type.SensorModel_jacobian_body}}
  }

{% endfor %}
  // clang-format on

}  // namespace {{namespace}}
