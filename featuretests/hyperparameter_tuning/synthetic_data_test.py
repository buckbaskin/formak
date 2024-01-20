"""
# Hyper-Parameter Tuning Feature Test.

Demonstrate tuning a model for two different innovation filtering hyper-parameters using the same process.
"""


import numpy as np
from formak.ui import DesignManager, StateId
from model import (
    calibration_map,
    process_noise,
    sensor_models,
    sensor_noises,
    symbolic_model,
)

from data import generate_data
from formak import ui

np.seterr("raise")


def test_with_synthetic_data():
    true_innovation = 5

    initial_state = DesignManager(name="mercury")

    symbolic_model_state = initial_state.symbolic_model(model=symbolic_model)

    fit_model_state = symbolic_model_state.fit_model(
        parameter_space={
            "process_noise": [process_noise],
            "sensor_models": [sensor_models],
            "sensor_noises": [sensor_noises],
            "calibration_map": [calibration_map],
            "innovation_filtering": [None, 1, 2, 3, 4, 5, 6, 7],
        },
        data=generate_data(true_innovation),
    )

    # Note: not a state transition
    python_model = fit_model_state.export_python()

    assert python_model.config.innovation_filtering is not None
    assert (
        true_innovation - 0.5
        < python_model.config.innovation_filtering
        < true_innovation + 0.5
    )


def test_state_machine_interface():
    initial_state = DesignManager(name="mercury")

    assert initial_state.history() == [StateId.Start]
    assert initial_state.available_transitions() == ["symbolic_model"]
    assert initial_state.search(StateId.Fit_Model) == ["symbolic_model", "fit_model"]

    symbolic_model_state = initial_state.symbolic_model(model=symbolic_model)

    assert symbolic_model_state.history() == [StateId.Start, StateId.Symbolic_Model]
    assert symbolic_model_state.available_transitions() == ["fit_model"]
    assert symbolic_model_state.search(StateId.Fit_Model) == ["fit_model"]
