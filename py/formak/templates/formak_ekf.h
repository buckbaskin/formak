#pragma once

namespace formak {

struct State {
  // clang-format off
  {{State_members}}
  // clang-format on
};

struct Covariance {
  // clang-format off
  {{Covariance_members}}
  // clang-format on
};

struct Control {
  // clang-format off
  {{Control_members}}
  // clang-format on
};

struct StateAndVariance {
  State state;
  Covariance covariance;
};

enum class SensorId {
  // clang-format off
    {{SensorId_members}}
  // clang-format on
};

template <SensorId Identifier, typename ReadingT>
struct SensorReading {
  SensorId id = Identifier;
  ReadingT reading;
};

class ExtendedKalmanFilter {
 public:
  StateAndVariance process_model(double dt, const StateAndVariance& input,
                                 const Control& input_control);

  template <SensorId Identifier, typename ReadingT>
  StateAndVariance sensor_model(
      const StateAndVariance& input,
      const SensorReading<Identifier, ReadingT>& input_reading) {
    return ReadingT::SensorModel::sensor_model(input, input_reading);
  }
};

}  // namespace formak
