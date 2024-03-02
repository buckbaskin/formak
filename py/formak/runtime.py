"""
Python Runtime.

A collection of classes and tools for running filters and providing additional functionality around the filter
"""

from bisect import bisect_left, bisect_right
from collections import namedtuple
from math import floor
from typing import Any, List, Optional


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


RollbackOptions = namedtuple(
    "RollbackOptions",
    ["max_history", "max_memory", "max_time", "time_resolution"],
    defaults=(None, None, None, 1e-9),
)
StorageLayout = namedtuple("StorageLayout", ["time", "state", "covariance", "sensors"])


class Storage:
    def __init__(self):
        self.data = []
        self.options = RollbackOptions()

    def store(
        self,
        time: float,
        state: Optional[Any],
        covariance: Optional[Any],
        sensors: Optional[Any],
    ):
        insertion_index = bisect_left(self.data, time, key=lambda e: e.time)
        print(
            f"inserting at {insertion_index} in range {insertion_index-1}:{insertion_index+1}",
            [e.time for e in self.data[insertion_index - 1 : insertion_index + 1]],
        )

        row = StorageLayout(
            time=time, state=state, covariance=covariance, sensors=sensors
        )

        # TODO: testing: insert before range, insert after range, insert middle
        # of range, insert w/ exact matching time, insert within time
        # resolution

        # TODO: test the following logic
        # Instead of blindly inserting, check if there's a time match.
        # If no match, insert
        # If match, update state, covariance, append sensors
        if len(self.data) > 0:
            candidate_time = round(
                self.data[insertion_index].time / self.options.time_resolution
            )
            insert_time = round(time / self.options.time_resolution)
            print("resolution match?", candidate_time, insert_time)

            if insert_time == candidate_time:
                self._update(insertion_index, row)
                return

        self._insert(insertion_index, row)

    def _insert(self, idx, row):
        self.data.insert(
            idx,
            row,
        )

    def load(self, time: float):
        """
        Load the latest time equal to or before the given time.
        If there are no entries before the given time, load the first entry.
        """
        assert isinstance(time, (float, int))
        # TODO: this might need to scan forwards or backwards to get a state or a time with a state, covariance?
        retrieval_index = bisect_left(self.data, time, key=lambda e: e.time)
        print(
            "retrieve target",
            time,
            "retrieval_index",
            retrieval_index,
            "data before insertion",
            [e.time for e in self.data],
        )
        retrieval_index = min(len(self.data) - 1, retrieval_index)
        return self.data[retrieval_index]

    def scan(self, start_time, end_time):
        # TODO: check these for off-by-ones
        start_index = bisect_left(self.data, start_time, key=lambda e: e.time)
        end_index = bisect_right(self.data, end_time, key=lambda e: e.time)

        yield from enumerate(self.data[start_index:end_index])
