#include <{{header_include}}>

// clang-format off
namespace {{namespace}} {
// clang-format-on
  State::State() : data(Eigen::Matrix<double, {{State_size}}, 1>::Zero()) {
  }
  State::State(const StateOptions& options)
      : {{State_options_constructor_initializer_list}} {
  }

  // clang-format off
{% if enable_control %}
  // clang-format on
  Control::Control()
      : data(Eigen::Matrix<double, {{Control_size}}, 1>::Zero()) {
  }
  Control::Control(const ControlOptions& options)
      : {{Control_options_constructor_initializer_list}} {}  // clang-format off
{% endif %}  // clang-format on

  // clang-format off
{% if enable_calibration %}
  // clang-format on
  Calibration::Calibration()
      : data(Eigen::Matrix<double, {{Calibration_size}}, 1>::Zero()) {
  }
  Calibration::Calibration(const CalibrationOptions& options)
      : {{Calibration_options_constructor_initializer_list}} {}
        // clang-format off
{% endif %}  // clang-format on

        State
        Model::model(
            double dt,
            const State& input_state
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
        ) {
    // clang-format off
{{Model_model_body}}
    // clang-format on
  }

}  // namespace {{namespace}}
