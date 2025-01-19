from __future__ import annotations

import sympy
from backend_py.config import Config
from backend_py.extended_kalman_filter import ExtendedKalmanFilter
from frontend.model_validation import model_validation
from problemdefinition.ui_model_base import UiModelBase
from sympy import Symbol


def compile_ekf(
    symbolic_model: UiModelBase,
    process_noise: dict[Symbol | tuple[Symbol, Symbol], float],
    sensor_models: dict[str, sympy.core.expr.Expr],
    sensor_noises,
    calibration_map: dict[Symbol, float] | None = None,
    *,
    config=None,
) -> ExtendedKalmanFilter:
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    model_validation(
        symbolic_model,
        process_noise,
        sensor_models,
        calibration_map=calibration_map,
        extra_validation=config.extra_validation,
    )

    return ExtendedKalmanFilter(
        state_model=symbolic_model,
        process_noise=process_noise,
        sensor_models=sensor_models,
        sensor_noises=sensor_noises,
        calibration_map=calibration_map,
        config=config,
    )
