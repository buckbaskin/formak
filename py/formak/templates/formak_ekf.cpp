#include <{{header_include}}>

namespace formak {

State Model::model(double dt, const State& input_state,
                   const Control& input_control) {
  // clang-format off
  {{Model_model_body}}
  // clang-format on
}

}  // namespace formak
