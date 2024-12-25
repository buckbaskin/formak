"""
Python Runtime.

A collection of classes and tools for running filters and providing additional functionality around the filter
"""

from collections import namedtuple
from math import floor
from typing import List, Optional


class StampedReading:
    def __init__(self, timestamp, sensor_key, *, _data=None, **kwargs):
        self.timestamp = timestamp
        self.sensor_key = sensor_key
        self._data = _data
        self.kwargs = kwargs

    @classmethod
    def from_data(cls, timestamp, sensor_key, data):
        return cls(_data=data)


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
        *,
        control=None,
        readings: Optional[List[StampedReading]] = None,
    ):
        """
        Update to a given output time, optionally updating with sensor readings.

        Note: name-only arguments required to prevent ambiguity in the case of
        calling the function with readings but no control
        """
        if control is None and self._impl.control_size > 0:
            raise TypeError(
                "TypeError: tick() missing 1 required positional argument: 'control'"
            )
        if readings is None:
            readings = []

        for sensor_reading in readings:
            assert isinstance(sensor_reading, StampedReading)

            self.current_time, (self.state, self.covariance) = self._process_model(
                sensor_reading.timestamp,
                control=control,
            )

            if sensor_reading._data is None:
                sensor_reading._data = self._impl.make_reading(
                    sensor_reading.sensor_key, **sensor_reading.kwargs
                )

            (self.state, self.covariance) = self._impl.sensor_model(
                state=self.state,
                covariance=self.covariance,
                sensor_key=sensor_reading.sensor_key,
                sensor_reading=sensor_reading._data,
            )

        _, state_and_variance = self._process_model(output_time, control)
        return state_and_variance

    def _process_model(self, output_time, control):
        # const
        max_dt = self._impl.config.max_dt_sec
        if self.current_time > output_time:
            max_dt = -0.1

        state = self.state
        covariance = self.covariance

        expected_iterations = abs(floor((output_time - self.current_time) / max_dt))

        for _ in range(expected_iterations):
            state, covariance = self._impl.process_model(
                max_dt, state, covariance, control
            )

        iter_time = self.current_time + max_dt * expected_iterations
        if abs(output_time - iter_time) >= 1e-9:
            state, covariance = self._impl.process_model(
                output_time - iter_time, state, covariance, control
            )

        return output_time, StateAndVariance(state, covariance)
