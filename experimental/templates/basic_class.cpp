#pragma once

namespace experimental {

class Stateful {
 public:
  Stateful() : _state(0.0) {
  }

  void update() {
    // clang-format off
    {{update_body}}
    // clang-format on
  }

  double getValue() {
    // clang-format off
    {{getValue_body}}
    // clang-format on
  }

 private:
  double _state;
};

class SympyModel {
 public:
  double model(double x, double y) {
    // clang-format off
    {{SympyModel_model_body}}
    // clang-format on
  }
};

}  // namespace experimental
