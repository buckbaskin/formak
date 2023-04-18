from formak import python
from model_definition import model_definition
import numpy as np


def test_python_EKF():
    model = model_definition()

    1 / 0

    orientation_rate_states, CON_orientation_rates_in_global_frame = named_rotation(
        "CON_orate"
    )
    acceleration_states = named_translation("CON_acc").free_symbols
    process_noise = {
        k: v
        for k, v in list(zip(orientation_rate_states, repeat(0.1)))
        + list(zip(acceleration_states, repeat(1.0)))
    }

    1 / 0

    python_implementation = python.compile_ekf(
        state_model=model,
        process_noise=process_noise,
        sensor_models={"simple": {ui.Symbol("v"): ui.Symbol("v")}},
        sensor_noises={"simple": np.eye(1)},
        config={"compile": True},
    )

    1 / 0

    state_vector = np.zeros((9, 1))
    control_vector = np.zeros((6, 1))

    1 / 0

    state_vector_next = python_implementation.model(0.01, state_vector, control_vector)
