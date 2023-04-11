from itertools import product

from formak.exceptions import ModelConstructionError
from sympy import Symbol, diff
from sympy.solvers.solveset import nonlinsolve


def model_validation(state_model, process_noise):
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
                f'Key {key} {type(key)} not in allow"list" [{render}]'
            )

    symbols_to_solve_for = list(state_model.state) + list(state_model.control)
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
        raise ModelConstructionError(
            f"Model has solutions in state space where covariance will collapse to zero. Example Solutions: {solutions[:3]}"
        )
