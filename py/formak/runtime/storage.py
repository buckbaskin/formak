"""
Python Runtime.

A collection of classes and tools for running filters and providing additional functionality around the filter
"""

from bisect import bisect_left, bisect_right
from collections import namedtuple
from typing import Any, List, Optional

from formak.runtime.common import RollbackOptions

StorageLayout = namedtuple(
    "StorageLayout", ["time", "state", "covariance", "control", "sensors"]
)


class Storage:
    def __init__(self, options=None):
        if options is None:
            options = RollbackOptions()

        self.options = options
        self.data = []

    def store(
        self,
        time: float,
        state: Optional[Any],
        covariance: Optional[Any],
        *,
        control: Optional[Any] = None,
        sensors: Optional[List[Any]] = None,
    ):
        time = round(time / self.options.time_resolution) * self.options.time_resolution

        insertion_index = bisect_left(self.data, time, key=lambda e: e.time)

        row = StorageLayout(
            time=time,
            state=state,
            covariance=covariance,
            control=control,
            sensors=list(sensors),
        )

        if len(self.data) > 0:
            # compare to last element if you would insert at the end of the list
            update_index = min(len(self.data) - 1, insertion_index)

            candidate_time_step = round(
                self.data[update_index].time / self.options.time_resolution
            )
            insert_time_step = round(time / self.options.time_resolution)

            if insert_time_step == candidate_time_step:
                self._update(update_index, row)
                return

        self._insert(insertion_index, row)

    def _insert(self, idx, row):
        self.data.insert(
            idx,
            row,
        )

    def _update(self, idx, row):
        existing_time = self.data[idx].time
        existing_sensors = self.data[idx].sensors
        existing_sensors.extend(row.sensors)

        new_control = self.data[idx].control
        if row.control is not None:
            new_control = row.control

        self.data[idx] = StorageLayout(
            time=existing_time,
            state=row.state,
            covariance=row.covariance,
            control=new_control,
            sensors=existing_sensors,
        )

    def load(self, time: float):
        """
        Load the latest time equal to or before the given time.
        If there are no entries before the given time, load the first entry.
        """
        assert isinstance(time, (float, int))
        retrieval_index = bisect_left(self.data, time, key=lambda e: e.time) - 1
        retrieval_index = max(0, min(retrieval_index, len(self.data) - 1))
        return self.data[retrieval_index]

    def scan(self, start_time=None, end_time=None):
        if (start_time is None) != (end_time is None):
            raise TypeError(
                "Storage.scan should be called with either both a start and end time or neither"
            )

        if start_time is None:
            yield from self.data

        else:
            controls_times = [(idx, e.time) for idx, e in enumerate(self.data) if e.control is not None]
            start_index_controls_times = bisect_left(controls_times, start_time, key=lambda e: e[1])
            start_index = controls_times[start_index_controls_times][0]
            print('scan', 'start bisect result', start_time, '->', start_index, [e.time for e in self.data[start_index-1:start_index+2]])

            end_index = bisect_right(self.data, end_time, key=lambda e: e.time)
            if end_index < len(self.data) and abs(self.data[end_index].time - end_time) <= self.options.time_resolution:
                end_index = end_index + 1
            print('scan', 'end bisect result', end_time, '->', end_index, [e.time for e in self.data[end_index-1:end_index+2]])

            yield from self.data[start_index:end_index]
