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

  State Model::model(double dt, const State& input_state,
                     const Control& input_control) {
    // clang-format off
{{Model_model_body}}
    // clang-format on
  }

}  // namespace {{namespace}}
