import pytest

from formak import python
from formak.ui import *


class Timer(object):
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.clock()

    def elapsed(self):
        return self.start - self.end


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

    model = Model(state=state, control=control, state_model=state_model)

    pure_implementation = python.compile(model, config={"compile": False})
    compiled_implementation = python.compile(model, config={"compile": True})

    state_vector = [0.0, 0.0, 0.0, 0.0]

    state_vector_next = pure_implementation.model(state_vector)
    state_vector_next_compiled = compiled_implementation.model(state_vector)

    assert state_vector_next == state_vector_next_compiled

    iters = 1000

    pure_timer = Timer()
    with pure_timer:
        for i in range(iters):
            state_vector_next = pure_implementation.model(state_vector_next)

    compiled_timer = Timer()
    with compiled_timer:
        for i in range(iters):
            state_vector_next_compiled = compiled_implementation.model(
                state_vector_next_compiled
            )

    assert compiled_timer.elapsed() < (pure_timer.elapsed() / 2.0)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_UI_simple())
