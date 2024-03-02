"""
Feature Test.

Run the rollback interface with some simple examples.

Passes if the rollback updates as expected the state
"""

from collections import namedtuple
from typing import Optional, List
from formak.runtime import ManagedFilter, StampedReading

RollbackOptions = namedtuple(
    "RollbackOptions", ["max_history", "max_memory", "max_time"]
)


class ManagedRollback:
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
        Returns (state, variance) tuple
        """
        # error handling (e.g. max time, max states, etc)

        # sort readings by time

        # load first state before first reading time

        # for each reading:
        #   process model to reading time
        #   sensor update at reading time
        #   save state after sensor update at reading time

        # process model to output time
        _, state_and_variance = self._process_model(output_time, control)
        return state_and_variance


class Illustrator:
    def process_model(self, dt, state, covariance, control):
        return state, covariance

    def sensor_model(self, state, covariance, sensor_key, sensor_reading):
        state.append(sensor_reading)
        return state, covariance


def test_rollback_empty():
    raise NotImplementedError("test_rollback_empty")


def test_rollback_basic_comparative():
    """
    Timeline:
    - Recieve A (on time)
    - Nothing
    - Recieve C (on time)
    - Recieve B (delayed), D (on time) [rollback]

    Expect:
    - [A, B, C, D]
    """

    rollback = ManagedRollback(
        ekf=Illustrator(), start_time=0, state=[], covariance=None
    )

    rollback.tick(1, readings=[StampedReading(1, "A")])
    rollback.tick(2, readings=[])
    rollback.tick(3, readings=[StampedReading(3, "C")])
    rollback_state, _ = rollback.tick(
        4, readings=[StampedReading(4, "D"), StampedReading(2, "B")]
    )

    assert rollback_state == ["A", "B", "C", "D"]

    mf = ManagedFilter(ekf=Illustrator(), start_time=0, state=[], covariance=None)

    mf.tick(1, readings=[StampedReading(1, "A")])
    mf.tick(2, readings=[])
    mf.tick(3, readings=[StampedReading(3, "C")])
    mf_state, _ = mf.tick(4, readings=[StampedReading(4, "D"), StampedReading(2, "B")])

    assert mf_state != ["A", "B", "C", "D"]
