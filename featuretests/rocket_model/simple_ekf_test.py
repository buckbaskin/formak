from formak import ui, python
from itertools import repeat
from model_definition import model_definition, named_rotation_rate, named_acceleration
import numpy as np


def test_python_EKF():
    model = model_definition()

    (reading_orientation_rate_states, _) = named_rotation_rate("IMU_reading")
    reading_acceleration_states = sorted(
        named_acceleration("IMU_reading").free_symbols, key=lambda x: x.name
    )

    process_noise = {
        k: v
        for k, v in list(zip(reading_orientation_rate_states, repeat(0.1)))
        + list(zip(reading_acceleration_states, repeat(1.0)))
    }

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
