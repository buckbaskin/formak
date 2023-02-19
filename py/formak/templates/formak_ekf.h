#pragma once

#include <cstddef>  // size_t

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

// clang-format off
{% for reading_type in reading_types %}
struct {{reading_type.typename}}SensorModel;

struct {{reading_type.typename}} {
  using SensorModel = {{reading_type.typename}}SensorModel;
  constexpr static size_t size = {{reading_type.size}};

  {{reading_type.members}}
};

struct {{reading_type.typename}}SensorModel {
    {{reading_type.typename}} sensor_model(
      const StateAndVariance& input,
      const SensorReading<{{reading_type.identifier}}, {{reading_type.typename}}>& input_reading);
};

{% endfor %}
// clang-format on

class ExtendedKalmanFilter {
 public:
  StateAndVariance process_model(double dt, const StateAndVariance& input,
                                 const Control& input_control);

  template <SensorId Identifier, typename ReadingT>
  StateAndVariance sensor_model(
      const StateAndVariance& input,
      const SensorReading<Identifier, ReadingT>& input_reading) {
    const State& state = input.state;
    const Covariance& covariance = input.covariance;

    const ReadingT& reading = input_reading.reading;
    const ReadingT predicted_reading =
        ReadingT::SensorModel::sensor_model(input, input_reading);

    // Here be the StateAndVariance math
    return input;
  }
};

}  // namespace formak
