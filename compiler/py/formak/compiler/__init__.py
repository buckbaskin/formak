import sympy
from formak import common
from sympy import Symbol
from formak.runtime.extended_kalman_filter import ExtendedKalmanFilter
from formak.compiler.config import Config


def compile_ekf(
    symbolic_model: common.UiModelBase,
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

    common.model_validation(
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
