from collections import namedtuple
from typing import List, Optional

StampedReading = namedtuple("StampedReading", ["timestamp", "sensor_key", "data"])

StateAndVariance = namedtuple("StateAndVariance", ["state", "covariance"])


class ManagedFilter:
    def __init__(self, ekf, start_time: float, state, covariance, calibration_map=None):
        self._impl = ekf
        self.current_time = start_time
        self.state = state
        self.covariance = covariance
        self.calibration_map = calibration_map

    def tick(
        self,
        output_time: float,
        control,
        readings: Optional[List[StampedReading]] = None,
    ):
        if readings is None:
            readings = []

        for sensor_reading in readings:
            assert isinstance(sensor_reading, StampedReading)

            self.current_time, (self.state, self.covariance) = self._process_model(
                sensor_reading.timestamp,
                control=control,
                calibration_map=calibration_map,
            )

            (self.state, self.covariance) = self._impl.sensor_model(
                sensor_reading.sensor_key,
                self.state,
                self.covariance,
                sensor_reading.data,
                calibration_map=calibration_map,
            )

        _, state_and_variance = self._process_model(output_time, control)
        return state_and_variance

    def _process_model(self, output_time, control):
        # const
        dt = 0.1
        current_time = self.current_time
        state = self.state
        covariance = self.covariance

        while current_time < output_time:
            current_time += dt
            state, covariance = self._impl.process_model(dt, state, covariance, control)

        return current_time, StateAndVariance(state, covariance)
