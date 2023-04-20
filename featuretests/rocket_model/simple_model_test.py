from formak import python
from model_definition import model_definition
import numpy as np


def test_python_Model():
    model = model_definition()

    python_implementation = python.compile(model)

    state_vector = np.zeros((9, 1))
    control_vector = np.zeros((6, 1))
    calibration_vector = np.zeros((6, 1))

    state_vector_next = python_implementation.model(
        0.01, state_vector, calibration_vector, control_vector
    )
