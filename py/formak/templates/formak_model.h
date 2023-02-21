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

  class Model {
   public:
    State model(double dt, const State& input_state,
                const Control& input_control);
  };

}  // namespace {{namespace}}
