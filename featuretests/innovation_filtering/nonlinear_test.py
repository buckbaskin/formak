"""
Innovation Filtering Featuretest

Create a model with states:
- x, y, heading, velocity model

Provide heading readings, expect rejecting 180 degree heading errors.
Nonlinear model provides clear divergence signal. If innovation filtering isn't
working as expected, then the model will flip into the wrong direction.
"""
from math import degrees, radians

import numpy as np
from sympy import cos, sin

from formak import python, runtime, ui

TRUE_SCALE = radians(5.0)


def make_ekf():
    dt = ui.Symbol("dt")

    x, y, heading, velocity, _heading_err = ui.symbols(
        ["x", "y", "heading", "velocity", "_heading_err"]
    )
    state = {x, y, heading}
    control = {velocity, _heading_err}

    state_model = {
        x: x + dt * velocity * cos(heading),
        y: y + dt * velocity * sin(heading),
        heading: heading + _heading_err,
    }

    model = ui.Model(dt=dt, state=state, control=control, state_model=state_model)

    config = python.Config(innovation_filtering=4)

    ekf = python.compile_ekf(
        state_model=model,
        process_noise={velocity: 1.0, _heading_err: 0.1},
        sensor_models={"compass": {heading: heading}},
        sensor_noises={"compass": TRUE_SCALE * np.eye(1)},
        config=config,
    )
    compass_model = python.SensorModel(model, {heading: heading}, {}, config)

    return ekf, compass_model.model


def test_obvious_innovation_rejections():
    ekf, compass_model = make_ekf()
    # Note: state = heading, x, y
    state = np.array([[0.0, 1.0, 0.0]]).transpose()
    covariance = np.eye(3)
    control = np.array([[0.0, 1.0]]).transpose()
    mf = runtime.ManagedFilter(
        ekf=ekf, start_time=0.0, state=state, covariance=covariance
    )

    readings = np.random.default_rng(seed=3).normal(
        loc=0.0, scale=TRUE_SCALE / 2.0, size=(100,)
    )

    readings[readings.shape[0] // 4] = radians(180.0)
    readings[readings.shape[0] // 3] = radians(180.0)
    readings[readings.shape[0] // 2] = radians(180.0)

    for idx, r in enumerate(readings):
        s = mf.tick(
            0.1 * idx,
            control=control,
            readings=[
                runtime.StampedReading(0.1 * idx - 0.05, "compass", np.array([[r]]))
            ],
        )

        if abs(compass_model(s.state)) >= TRUE_SCALE * 4:
            print({"idx": idx, "reading": degrees(r)})
        assert abs(compass_model(s.state)) < TRUE_SCALE * 4
