#pragma once

namespace formak {

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

}  // namespace formak
