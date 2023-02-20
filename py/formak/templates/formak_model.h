#pragma once

// clang-format off
namespace {{namespace}} {
// clang-format-on
  struct State {
    // clang-format off
  {{State_members}}
    // clang-format on
  };

  struct Control {
    // clang-format off
  {{Control_members}}
    // clang-format on
  };

  class Model {
   public:
    State model(double dt, const State& input_state,
                const Control& input_control);
  };

}  // namespace {{namespace}}
