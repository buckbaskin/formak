from __future__ import annotations

import numpy as np
from formak.exceptions import ModelConstructionError

from formak.compiler.basic_block import BasicBlock
from formak.common.named_vector import named_vector
from formak.common.named_covariance import named_covariance


class SensorModel:
    def __init__(self, state_model, sensor_model, calibration_map, config):
        self.readings = sorted(list(sensor_model.keys()))
        self.sensor_models = sensor_model

        self.sensor_size = len(self.readings)
        self.state_size = len(state_model.state)
        self.calibration_size = len(state_model.calibration)

        self.arglist_state = sorted(list(state_model.state), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(state_model.calibration), key=lambda x: x.name
        )
        self.arglist = self.arglist_state + self.arglist_calibration

        self.State = named_vector("State", self.arglist_state)
        self.Covariance = named_covariance("Covariance", self.arglist_state)
        self.Calibration = named_vector("Calibration", self.arglist_calibration)
        self.Reading = named_vector("Reading", self.readings)
        self.ReadingCovariance = named_vector("ReadingCovariance", self.readings)

        self.calibration_vector = np.array(
            [[calibration_map[k] for k in self.arglist_calibration]]
        ).transpose()
        if self.calibration_vector.shape != (self.calibration_size, 1):
            raise ModelConstructionError(
                f"calibration vector shape {self.calibration_vector.shape} doesn't match expected shape {(self.calibration_size, 1)}"
            )

        self._impl = BasicBlock(
            arglist=self.arglist,
            statements=[sensor_model[k] for k in self.readings],
            config=config,
        )

        ## "Pre-flight" Checks

        # Pre-check model for type errors
        self.model(self.State())

    def __len__(self):
        return len(self.sensor_models)

    def model(self, state_vector):
        assert isinstance(state_vector, self.State)

        reading = np.zeros((self.sensor_size, 1))
        for i, (reading_id, result) in enumerate(
            zip(
                self.readings,
                self._impl.execute(*state_vector, *self.calibration_vector),
            )
        ):
            try:
                reading[i, 0] = result
            except (TypeError, ValueError):
                print(
                    "Error when trying to process sensor model for reading %s"
                    % (reading_id,)
                )
                print("expected: float")
                print("given: {}, {}".format(state_vector, self.calibration_vector))
                if "result" in locals():
                    print("found: {}, {}".format(type(result), result))
                raise
        return self.Reading.from_data(reading)
