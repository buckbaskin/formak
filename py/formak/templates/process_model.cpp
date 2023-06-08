const Covariance& covariance = input.covariance;
// G = process_jacobian
ExtendedKalmanFilter::ProcessJacobianT G =
    ExtendedKalmanFilter::ProcessModel::process_jacobian(
        dt,
        input
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        input_calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        input_control
        // clang-format off
{% endif %}  // clang-format on
    );
// V = control_jacobian
ExtendedKalmanFilter::ControlJacobianT V =
    ExtendedKalmanFilter::ProcessModel::control_jacobian(
        dt,
        input
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        input_calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        input_control
        // clang-format off
{% endif %}  // clang-format on
    );
// M = process_noise
ExtendedKalmanFilter::CovarianceT M =
    ExtendedKalmanFilter::ProcessModel::covariance(
        dt,
        input
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        input_calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        input_control
        // clang-format off
{% endif %}  // clang-format on
    );

// Update State Estimate
// next_state = process_model(input, input_control)
State next_state = ExtendedKalmanFilter::ProcessModel::model(
    dt,
    input
    // clang-format off
{% if enable_calibration %}
    // clang-format on
    ,
    input_calibration
    // clang-format off
{% endif %}  // clang-format on
    // clang-format off
{% if enable_control %}
    // clang-format on
    ,
    input_control
    // clang-format off
{% endif %}  // clang-format on
);

// Update Covariance
// Sigma = G * Sigma * G.T + V * M * V.T
Covariance next_covariance;
next_covariance.data =
    G * covariance.data * G.transpose() + V * M * V.transpose();

return {.state = next_state, .covariance = next_covariance};
