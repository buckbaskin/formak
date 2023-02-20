#include <{{header_include}}>

// clang-format off
namespace {{namespace}} {
// clang-format-on
  State Model::model(double dt, const State& input_state,
                     const Control& input_control) {
    // clang-format off
  {{Model_model_body}}
    // clang-format on
  }

}  // namespace {{namespace}}
