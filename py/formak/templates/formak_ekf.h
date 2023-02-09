#pragma once

namespace formak {

struct State {
  // clang-format off
  {{State_members}}
  // clang-format on
};

struct StateVariance {
  // clang-format off
  {{StateVariance_members}}
  // clang-format on
};

struct Control {
  // clang-format off
  {{Control_members}}
  // clang-format on
};

struct StateAndVariance {
  State state;
  StateVariance covariance;
};

enum class SensorId {
  // clang-format off
    {{SensorId_members}}
  // clang-format on
};

template <size_t ReadingT>
struct SensorReading {
  SensorId id;
  ReadingT reading;
};

class ExtendedKalmanFilter {
 public:
  StateAndVariance process_model(double dt, const StateAndVariance& input_state,
                                 const Control& input_control);

  template <size_t ReadingT>
  StateAndVariance sensor_model(const StateAndVariance& input_state,
                                const SensorReading<ReadingT>& input_reading);
};

}  // namespace formak
