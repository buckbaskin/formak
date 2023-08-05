const Covariance& Sigma = state.covariance;
// G = process_jacobian
ExtendedKalmanFilter::ProcessJacobianT G =
    ExtendedKalmanFilter::ProcessModel::process_jacobian(
        dt,
        state
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        control
        // clang-format off
{% endif %}  // clang-format on
    );
// V = control_jacobian
ExtendedKalmanFilter::ControlJacobianT V =
    ExtendedKalmanFilter::ProcessModel::control_jacobian(
        dt,
        state
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        control
        // clang-format off
{% endif %}  // clang-format on
    );
// M = process_noise
ExtendedKalmanFilter::CovarianceT M =
    ExtendedKalmanFilter::ProcessModel::covariance(
        dt,
        state
        // clang-format off
{% if enable_calibration %}
        // clang-format on
        ,
        calibration
        // clang-format off
{% endif %}  // clang-format on
        // clang-format off
{% if enable_control %}
        // clang-format on
        ,
        control
        // clang-format off
{% endif %}  // clang-format on
    );

// Update State Estimate
// next_state = process_model(state, control)
State next_state = ExtendedKalmanFilter::ProcessModel::model(
    dt,
    state
    // clang-format off
{% if enable_calibration %}
    // clang-format on
    ,
    calibration
    // clang-format off
{% endif %}  // clang-format on
    // clang-format off
{% if enable_control %}
    // clang-format on
    ,
    control
    // clang-format off
{% endif %}  // clang-format on
);

// Update Covariance
// Sigma = G * Sigma * G.T + V * M * V.T
Covariance next_covariance;
next_covariance.data = G * Sigma.data * G.transpose() + V * M * V.transpose();

return {.state = next_state, .covariance = next_covariance};
