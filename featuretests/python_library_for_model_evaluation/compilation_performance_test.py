import numpy as np
import pytest

from datetime import datetime, timedelta

from formak import python
from formak.ui import *


class Timer(object):
    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self.end = datetime.now()

    def elapsed(self):
        return self.end - self.start


def test_UI_simple():
    dt = Symbol("dt")

    tp = trajectory_properties = {k: Symbol(k) for k in ["mass", "z", "v", "a"]}

    thrust = Symbol("thrust")

    state = set(tp.values())
    control = set([thrust])

    state_model = {
        tp["mass"]: tp["mass"],
        tp["z"]: tp["z"] + dt * tp["v"],
        tp["v"]: tp["v"] + dt * tp["a"],
        tp["a"]: -9.81 * tp["mass"] + thrust,
    }

    model = Model(dt=dt, state=state, control=control, state_model=state_model)

    pure_implementation = python.compile(model, config={"compile": False})
    compiled_implementation = python.compile(model, config={"compile": True})

    state_vector = np.array([[0.0, 0.0, 0.0, 0.0]]).transpose()
    control_vector = np.array([[0.0]])

    state_vector_next = pure_implementation.model(0.1, state_vector, control_vector)
    state_vector_next_compiled = compiled_implementation.model(
        0.1, state_vector, control_vector
    )

    assert (state_vector_next == state_vector_next_compiled).all()

    iters = 100000

    pure_timer = Timer()
    with pure_timer:
        for i in range(iters):
            state_vector_next = pure_implementation.model(
                0.1, state_vector_next, control_vector
            )

    compiled_timer = Timer()
    with compiled_timer:
        for i in range(iters):
            state_vector_next_compiled = compiled_implementation.model(
                0.1, state_vector_next_compiled, control_vector
            )

    assert pure_timer.elapsed() > timedelta(seconds=0.1)
    assert compiled_timer.elapsed() < (pure_timer.elapsed() / 2.0)
