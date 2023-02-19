#include <{{header_include}}>

namespace formak {

StateAndVariance ExtendedKalmanFilter::process_model(
    double dt, const StateAndVariance& input, const Control& input_control){
    // clang-format off
  {{ExtendedKalmanFilter_process_model_body}}
    // clang-format on
}

// clang-format off
{% for reading_type in reading_types %}
{{reading_type.typename}} {{reading_type.typename}}SensorModel::sensor_model(
      const StateAndVariance& input,
      const SensorReading<{{reading_type.identifier}}, {{reading_type.typename}}>& input_reading) {
  {{reading_type.SensorModel_sensor_model_body}}
}

{% endfor %}
// clang-format on

}  // namespace formak
