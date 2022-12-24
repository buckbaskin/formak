#pragma once

namespace experimental {

class Stateful {
 public:
  Stateful() : _state(0.0) {
  }

  void update() {
    // TODO(buck): clang-format off
    {
      {}
    }
    // TODO(buck): clang-format on
  }

  double getValue() {
    // TODO(buck): clang-format off
    {
      {}
    }
    // TODO(buck): clang-format on
  }

 private:
  double _state;
};

}  // namespace experimental
