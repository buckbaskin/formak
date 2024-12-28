from __future__ import annotations

import numpy as np
from formak.exceptions import ModelConstructionError

from formak import common
from formak.compiler.config import Config
from formak.compiler.basic_block import BasicBlock


class Model:
    """Python implementation of the model."""

    def __init__(self, symbolic_model, config, calibration_map=None):
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)

        if calibration_map is None:
            calibration_map = {}

        self.state_size = len(symbolic_model.state)
        self.calibration_size = len(symbolic_model.calibration)
        self.control_size = len(symbolic_model.control)

        self.arglist_state = sorted(list(symbolic_model.state), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(symbolic_model.calibration), key=lambda x: x.name
        )
        self.arglist_control = sorted(
            list(symbolic_model.control), key=lambda x: x.name
        )
        self.arglist = (
            [symbolic_model.dt]
            + self.arglist_state
            + self.arglist_calibration
            + self.arglist_control
        )

        self.State = common.named_vector("State", self.arglist_state)
        self.Control = common.named_vector("Control", self.arglist_control)
        self.Calibration = common.named_vector("Calibration", self.arglist_calibration)

        self.calibration_vector = np.zeros((0, 0))
        if self.calibration_size > 0:
            if len(calibration_map) == 0:
                map_lite = ", ".join(
                    [f"{k}: ..." for k in self.arglist_calibration[:3]]
                )
                if len(self.arglist_calibration) > 3:
                    map_lite += ", ..."
                raise ModelConstructionError(
                    f"Model Missing specification of calibration_map: {{{map_lite}}}"
                )
            if len(calibration_map) != self.calibration_size:
                missing_from_map = set(symbolic_model.calibration) - set(
                    calibration_map.keys()
                )
                extra_from_map = set(calibration_map.keys()) - set(
                    symbolic_model.calibration
                )
                missing = ""
                if len(missing_from_map) > 0:
                    missing = f"\nMissing: {missing_from_map}"
                extra = ""
                if len(extra_from_map) > 0:
                    extra = f"\nExtra: {extra_from_map}"
                raise ModelConstructionError(f"Mismatched Calibration:{missing}{extra}")
        self.calibration_vector = np.array(
            [[calibration_map[k] for k in self.arglist_calibration]]
        ).transpose()
        if self.calibration_vector.shape != (self.calibration_size, 1):
            raise ModelConstructionError(
                f"calibration_vector shape {self.calibration_vector.shape} doesn't match {(self.calibration_size, 1)}"
            )

        self._impl = BasicBlock(
            arglist=self.arglist,
            statements=[symbolic_model.state_model[a] for a in self.arglist_state],
            config=config,
        )

    def model(self, dt, state, control=None):
        if control is None:
            if self.control_size > 0:
                raise TypeError(
                    "model() missing 1 required positional argument: 'control'"
                )
            control = self.Control()

        try:
            assert isinstance(dt, float)
            assert isinstance(state, self.State)
            assert isinstance(control, self.Control)
        except AssertionError:
            print(
                f"model({type(dt)} {dt}, {type(state)} {state}, {type(control)} {control}"
            )
            raise

        next_state = self.State(
            **{
                str(state_id): result
                for state_id, result in zip(
                    self.arglist_state,
                    self._impl.execute(dt, *state, *self.calibration_vector, *control),
                )
            }
        )

        return next_state


def compile(symbolic_model, calibration_map=None, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    common.model_validation(
        symbolic_model,
        {},
        {},
        calibration_map=calibration_map,
        extra_validation=config.extra_validation,
    )

    return Model(
        symbolic_model=symbolic_model, calibration_map=calibration_map, config=config
    )
