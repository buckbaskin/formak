#pragma once

#include <Eigen/Dense>  // Matrix

// clang-format off
namespace {{namespace}} {
// clang-format-on
  struct StateOptions {
    // clang-format off
    {{ StateOptions_members }}
    // clang-format on
  };

  struct State {
    State();
    State(const StateOptions& options);
    // clang-format off
    {{State_members}}
    // clang-format on
    Eigen::Matrix<double, {{State_size}}, 1> data =
        Eigen::Matrix<double, {{State_size}}, 1>::Zero();
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

  class Model {
   public:
    State model(
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
    );
  };

}  // namespace {{namespace}}
