"""
# Hyper-Parameter Tuning Feature Test.

Demonstrate tuning a model for two different innovation filtering hyper-parameters using the same process.
"""


from formak.ui import DesignManager
from model import symbolic_model

from data import generate_data
from formak import ui


def test_with_synthetic_data():
    true_innovation = 5

    initial_state = DesignManager(name="mercury")

    # Q: No-discard but for python?
    symbolic_model_state = initial_state.symbolic_model(model=symbolic_model)

    fit_model_state = symbolic_model_state.fit_model(
        parameter_space={}, data=generate_data(true_innovation)
    )

    # Note: not a state transition
    python_model = fit_model_state.export_python()

    assert (
        true_innovation - 0.5
        < python_model.config.innovation_filtering
        < true_innovation + 0.5
    )


def test_state_machine_interface():
    initial_state = DesignManager(name="mercury")

    # TODO: make the state names enums
    assert initial_state.history() == ["Start"]
    assert initial_state.available_transitions() == ["symbolic_model"]
    assert initial_state.search("Fit Model") == ["symbolic_model", "fit_model"]

    # Q: No-discard but for python?
    symbolic_model_state = initial_state.symbolic_model(model=symbolic_model)

    assert symbolic_model_state.history() == ["Start", "Symbolic Model"]
    assert symbolic_model_state.available_transitions() == ["fit_model"]
    assert symbolic_model_state.search("Fit Model") == ["fit_model"]
