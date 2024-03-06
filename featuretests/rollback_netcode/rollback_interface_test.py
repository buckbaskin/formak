"""
Feature Test.

Run the rollback interface with some simple examples.

Passes if the rollback updates as expected the state
"""

from dataclasses import dataclass
from typing import Iterable

from formak.python import Config
from formak.runtime import ManagedFilter, ManagedRollback, StampedReading
from numpy.testing import assert_allclose


@dataclass
class IllustratorState:
    time: float
    queue: Iterable[str]


class Illustrator:
    config = Config()
    control_size = 0

    def process_model(self, dt, state, covariance, control):
        result = IllustratorState(state.time + dt, state.queue)
        print("    - process_model", dt, " -> ", result.time)
        return result, covariance

    def sensor_model(self, state, covariance, sensor_key, sensor_reading):
        result = IllustratorState(state.time, state.queue + (sensor_reading,))
        return result, covariance

    def make_reading(self, sensor_key, **kwargs):
        return sensor_key


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
        ekf=Illustrator(),
        start_time=0,
        state=IllustratorState(0, tuple()),
        covariance=None,
    )

    rollback_state, _ = rollback.tick(1, readings=[StampedReading(1, "A")])
    assert rollback_state.queue == ("A",)

    rollback_state, _ = rollback.tick(2, readings=[])
    assert rollback_state.queue == ("A",)

    rollback_state, _ = rollback.tick(3, readings=[StampedReading(3, "C")])
    assert rollback_state.queue == (
        "A",
        "C",
    )

    rollback_state, _ = rollback.tick(
        4, readings=[StampedReading(4, "D"), StampedReading(2, "B")]
    )
    assert_allclose(rollback_state.time, 4)
    assert rollback_state.queue == ("A", "B", "C", "D")

    mf = ManagedFilter(
        ekf=Illustrator(),
        start_time=0,
        state=IllustratorState(0, tuple()),
        covariance=None,
    )

    mf.tick(1, readings=[StampedReading(1, "A")])
    mf.tick(2, readings=[])
    mf.tick(3, readings=[StampedReading(3, "C")])
    mf_state, _ = mf.tick(4, readings=[StampedReading(4, "D"), StampedReading(2, "B")])

    assert_allclose(mf_state.time, 4)
    assert mf_state.queue != ["A", "B", "C", "D"]
