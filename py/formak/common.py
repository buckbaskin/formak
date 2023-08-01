from itertools import product
from typing import Dict

from formak.exceptions import ModelConstructionError
from sympy import Symbol, diff
from sympy.solvers.solveset import nonlinsolve


def model_validation(
    state_model,
    process_noise,
    sensor_models,
    *,
    verbose=True,
    extra_validation=False,
    calibration_map: Dict[Symbol, float],
):
    assert isinstance(process_noise, dict)
    allowed_keys = set(
        list(state_model.control)
        + [
            (x, y)
            for x, y in product(state_model.control, state_model.control)
            if x != y
        ]
    )
    for key in process_noise:
        if not isinstance(key, Symbol):
            raise ModelConstructionError(
                f"Key {key} needs to be type Symbol, found type {type(key)}"
            )
        if key not in allowed_keys:
            render = ", ".join([f"{k} {type(k)}" for k in allowed_keys])
            raise ModelConstructionError(
                f'Key {key} {type(key)} not in allow"list" of keys and combinations for process noise based on state_model.control [{render}]'
            )
    if set(calibration_map.keys()) != state_model.calibration:
        map_version = set(calibration_map.keys())
        missing_calibrations = state_model.calibration - map_version
        missing_calibrations = ", ".join(
            sorted([symbol.name for symbol in missing_calibrations])[:3]
        )
        extra_mappings = map_version - state_model.calibration
        extra_mappings = ", ".join(
            sorted([symbol.name for symbol in extra_mappings])[:3]
        )
        raise ModelConstructionError(
            f"Mismatch in Model calibration: Missing from map? {missing_calibrations} | Missing from setup? {extra_mappings}"
        )

    # Check if sensor models depend on values outside the state, calibration [and map]
    allowed_symbols = set(state_model.state) | set(state_model.calibration)
    for k, model_set in sensor_models.items():
        for k2, model in model_set.items():
            if not set(model.free_symbols).issubset(allowed_symbols):
                extra_symbols = sorted(list(set(model.free_symbols) - allowed_symbols))
                raise ModelConstructionError(
                    f"Sensor Model[{k}][{k2}] has symbols not in state, calibration: {extra_symbols}"
                )

    # Note: flagging this off for now because it was running into performance issues
    if extra_validation:
        symbols_to_solve_for = list(state_model.state)
        equations_to_solve = [
            diff(model, symbol)
            for model, symbol in product(
                state_model.state_model.values(), state_model.state
            )
        ]
        results_set = nonlinsolve(equations_to_solve, symbols_to_solve_for)
        if len(results_set) > 0:
            solutions = [
                dict(zip(symbols_to_solve_for, solution))
                for solution in sorted(list(results_set))
            ]
            solution_repr = "\n - ".join(map(lambda x: str(x), solutions[:3]))
            if verbose:
                print("symbols_to_solve_for")
                print(symbols_to_solve_for)
                print("equations_to_solve")
                for e in equations_to_solve:
                    print(f" - {e}")
            raise ModelConstructionError(
                f"Model has solutions in state space where covariance will collapse to zero. Example Solutions:\n -{solution_repr}"
            )
