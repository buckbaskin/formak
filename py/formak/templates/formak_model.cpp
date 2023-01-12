#include <{{header_include}}>

namespace experimental {

void Stateful::update() {
  // clang-format off
  {{update_body}}
  // clang-format on
}

double SympyModel::model(double x, double y) {
  // clang-format off
  {{SympyModel_model_body}}
  // clang-format on
}

}  // namespace experimental
