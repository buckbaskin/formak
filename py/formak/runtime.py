from collections import namedtuple
from typing import List, Optional

StampedReading = namedtuple("StampedReading", ["timestamp", "data"])


class ManagedFilter(object):
    def __init__(self, ekf, start_time: float, state, covariance):
        self._impl = ekf
        self.current_time = start_time
        self.state = state
        self.covariance = covariance

    def tick(self, output_time: float, readings: Optional[List[StampedReading]] = None):
        if readings is None:
            readings = []

        for sensor_reading in readings:
            assert isinstance(sensor_reading, StampedReading)

            self.current_time, (self.state, self.covariance) = self._process_model(
                sensor_reading.timestamp, self.state, self.covariance
            )

            (self.state, self.covariance) = self._impl.sensor_model(
                sensor_reading.data, self.state, self.covariance
            )

        return self._process_model(output_time, control)

    def _process_model(self, target_time, state, covariance):
        # const
        raise NotImplementedError("_process_model")

    # typename Impl::StateAndVarianceT tick(
    #     double outputTime, const typename Impl::ControlT& control) {
    #   return processUpdate(outputTime, control);
    # }
    # typename Impl::StateAndVarianceT tick(
    #     double outputTime, const typename Impl::ControlT& control,
    #     const std::vector<StampedReading>& readings) {
    #   for (const auto& stampedReading : readings) {
    #     _state = processUpdate(stampedReading.timestamp, control);
    #     _currentTime = stampedReading.timestamp;

    #     _state = stampedReading.data->sensor_model(_impl, _state);
    #   }

    #   return tick(outputTime, control);
    # }

    # private:
    # typename Impl::StateAndVarianceT processUpdate(
    #     double outputTime, const typename Impl::ControlT& control) const {
    #   double dt = 0.1;
    #   typename Impl::StateAndVarianceT state = _state;
    #   for (double currentTime = _currentTime; currentTime < outputTime;
    #        currentTime += dt) {
    #     state = _impl.process_model(dt, state, control);
    #   }
    #   return state;
    # }
