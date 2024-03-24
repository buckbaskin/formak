"""
ManagedRollback

"""
from math import floor
from typing import List, Optional

from formak.runtime.common import StampedReading, StateAndVariance
from formak.runtime.storage import Storage


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
        print("init")
        self._impl = ekf
        self.storage = storage if storage is not None else Storage()

        print('init', 'state', state)
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
        print("tick")

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

        self.current_time, self.state, self.covariance, control, _ = self.storage.load(
            start_time
        )
        print('tick', 'state', 'after load', self.state)
        print('tick', 'control', 'after load', control)

        # implicitly sorts readings by time introducing them into the global state queue
        # TODO: check if this is doing the correct thing, I don't want to overwrite a previously stored controls for this line
        for reading in readings:
            self.storage.store(
                reading.timestamp, state=None, covariance=None, control=None, sensors=[reading]
            )

        # for each reading:
        #   process model to reading time
        #   sensor update at reading time for all sensor readings
        #   save state after sensor update at reading time
        for sensor_time, _, _, control, sensors in self.storage.scan(start_time, output_time):
            self.current_time, (self.state, self.covariance) = self._process_model(
                sensor_time,
                control=control,
            )

            for sensor_reading in sensors:
                if sensor_reading._data is None:
                    sensor_reading._data = self._impl.make_reading(
                        sensor_reading.sensor_key, **sensor_reading.kwargs
                    )

                print("    - sensor update", sensor_reading.sensor_key)
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
                control=control,
                # sensors already stored, so don't pass new data
                sensors=[],
            )

        # process model to output time, don't store
        print('_process_model to output time', output_time, 'from', self.current_time)
        print('input: ', self.state)
        _, state_and_variance = self._process_model(output_time, control)
        print('output: ', state_and_variance.state)
        return state_and_variance

    def _process_model(self, output_time, control):
        # const
        max_dt = self._impl.config.max_dt_sec
        if self.current_time > output_time:
            max_dt = -self._impl.config.max_dt_sec

        state = self.state
        covariance = self.covariance

        expected_iterations = abs(floor((output_time - self.current_time) / max_dt))
        print('_process_model', 'expected_iterations', expected_iterations)

        for _ in range(expected_iterations):
            state, covariance = self._impl.process_model(
                max_dt, state, covariance, control
            )

        iter_time = self.current_time + max_dt * expected_iterations
        print('_process_model', 'iter_time', iter_time)
        if abs(output_time - iter_time) >= 1e-9:
            print('_process_model', 'process_model', 'before', state, output_time - iter_time)
            print('_process_model', 'process_model', 'before', 'control', control)
            state, covariance = self._impl.process_model(
                output_time - iter_time, state, covariance, control
            )
            print('_process_model', 'process_model', 'after', state)

        return output_time, StateAndVariance(state, covariance)
