#pragma once

namespace experimental {

class Stateful {
 public:
  Stateful() : _state(0.0) {
  }

  void update();

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
  double model(double x, double y);
};

}  // namespace experimental
