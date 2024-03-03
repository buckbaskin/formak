"""
Feature Test.

Run the rollback interface with some simple examples.

Passes if the rollback updates as expected the state
"""

from dataclasses import dataclass
from math import floor
from typing import List, Optional, Iterable

from formak.python import Config
from formak.runtime import (
    ManagedFilter,
    StampedReading,
    StateAndVariance,
    Storage,
)


class ManagedRollback:
    def __init__(
        self,
        ekf,
        start_time: float,
        state,
        covariance,
        calibration_map=None,
        *,
        storage=None,
    ):
        self._impl = ekf
        self.storage = storage if storage is not None else Storage()

        self.storage.store(start_time, state, covariance, sensors=[])
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
        # TODO: error handling (e.g. max time, max states, etc)

        if control is None and self._impl.control_size > 0:
            raise TypeError(
                "TypeError: tick() missing 1 required positional argument: 'control'"
            )
        if readings is None:
            readings = []

        # if no readings, then just process forward from the last state before the output time
        start_time = output_time
        if len(readings) > 0:
            start_time = min(readings, key=lambda r: r.timestamp).timestamp

        # TODO: test the following
        # Important: **before** inserting the current readings to process, load
        # the state at/before the first reading. That way you don't
        # accidentally load the state that was inserted from the first reading

        # load first state before first reading time
        # ignore sensors, they're already included in the state-covariance
        self.current_time, self.state, self.covariance, _ = self.storage.load(
            start_time
        )

        # implicitly sorts readings by time introducing them into the global state queue
        for reading in readings:
            self.storage.store(
                reading.timestamp, state=None, covariance=None, sensors=[reading]
            )

        # for each reading:
        #   process model to reading time
        #   sensor update at reading time for all sensor readings
        #   save state after sensor update at reading time
        for idx, (sensor_time, _, _, sensors) in self.storage.scan(
            start_time, output_time
        ):
            self.current_time, (self.state, self.covariance) = self._process_model(
                sensor_time,
                control=control,
            )

            for sensor_reading in sensors:
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

            self.storage.store(
                time=sensor_time,
                state=self.state,
                covariance=self.covariance,
                sensors=sensors,
            )

        # process model to output time
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


@dataclass
class IllustratorState:
    time: float
    queue: Iterable[str]


class Illustrator:
    config = Config()
    control_size = 0

    def process_model(self, dt, state, covariance, control):
        state.time += dt
        return state, covariance

    def sensor_model(self, state, covariance, sensor_key, sensor_reading):
        result = IllustratorState(state.time, state.queue + (sensor_reading,))
        return result, covariance

    def make_reading(self, sensor_key, **kwargs):
        return sensor_key


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
        ekf=Illustrator(), start_time=0, state=IllustratorState(0, tuple()), covariance=None
    )

    rollback_state, _ = rollback.tick(1, readings=[StampedReading(1, "A")])
    assert rollback_state.queue == ("A",)

    rollback_state, _ = rollback.tick(2, readings=[])
    assert rollback_state.queue == ("A",)

    rollback_state, _ = rollback.tick(3, readings=[StampedReading(3, "C")])
    assert rollback_state.queue == ("A", "C",)

    rollback_state, _ = rollback.tick(
        4, readings=[StampedReading(4, "D"), StampedReading(2, "B")]
    )
    assert rollback_state.queue == ("A", "B", "C", "D")

    mf = ManagedFilter(ekf=Illustrator(), start_time=0, state=[], covariance=None)

    mf.tick(1, readings=[StampedReading(1, "A")])
    mf.tick(2, readings=[])
    mf.tick(3, readings=[StampedReading(3, "C")])
    mf_state, _ = mf.tick(
        4, readings=[StampedReading(4, "D"), StampedReading(2, "B")]
    )

    assert mf_state.queue != ["A", "B", "C", "D"]
