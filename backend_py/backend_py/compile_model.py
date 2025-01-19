from __future__ import annotations

from backend_py.config import Config
from backend_py.model import Model
from frontend.model_validation import model_validation


def compile_model(symbolic_model, calibration_map=None, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    model_validation(
        symbolic_model,
        {},
        {},
        calibration_map=calibration_map,
        extra_validation=config.extra_validation,
    )

    return Model(
        symbolic_model=symbolic_model, calibration_map=calibration_map, config=config
    )
