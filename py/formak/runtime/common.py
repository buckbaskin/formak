"""
Runtime Common.

Data structures for other runtime classes and algorithms
"""

from collections import namedtuple


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


RollbackOptions = namedtuple(
    "RollbackOptions",
    ["max_history", "max_memory", "max_time", "time_resolution"],
    defaults=(None, None, None, 1e-9),
)
