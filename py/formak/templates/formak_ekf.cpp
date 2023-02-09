#include <{{header_include}}>

namespace formak {

StateAndVariance ExtendedKalmanFilter::process_model(
    double dt, const StateAndVariance& input_state,
    const Control& input_control) {
  // clang-format off
  {{ExtendedKalmanFilter_process_model_body}}
  // clang-format on
}

}  // namespace formak
