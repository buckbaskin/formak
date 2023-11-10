"""
# Hyper-Parameter Tuning Feature Test

Demonstrate tuning a model for two different innovation filtering hyper-parameters using the same process.
"""

from formak.ui import DManager
from model import symbolic_model

from data import generate_data


def test_with_synthetic_data():
    true_innovation = 5

    initial_state = DManager(name="mercury")

    # Q: No-discard but for python?
    symbolic_model_state = initial_state.symbolic_model(model=symbolic_model)

    fit_model_state = symbolic_model_state.fit_model(
        params={}, data=generate_data(true_innovation)
    )

    # Note: not a state transition
    python_model = fit_model_state.export_python()

    assert (
        true_innovation - 0.5
        < python_model.config.innovation_filtering
        < true_innovation + 0.5
    )
