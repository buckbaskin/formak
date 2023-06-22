from datetime import datetime

import numpy as np
import pytest
from formak.ui import *
from sympy import cos, sin, symbols

from formak import python


def combine_nodes(leaves):
    for i in range(0, len(leaves) // 2):
        l = leaves[i * 2]
        r = leaves[i * 2 + 1]
        yield 2 * sin(l * r) * cos(l + r)


def test_python_CSE():
    l, r = symbols(["l", "r"])

    # 2 * sin(l * r) * cos(l + r)

    leaves_count = 1024

    leaves = [l, r] * leaves_count

    nodes = list(combine_nodes(leaves))

    while len(nodes) > 1:
        nodes = list(combine_nodes(nodes))

    symbolic_model = nodes[0]

    dt = Symbol("dt")

    state = {l, r}
    state_model = {
        l: symbolic_model,
        r: symbolic_model,
    }
    control = {}

    model = Model(dt=dt, state=state, control=control, state_model=state_model)

    cse_implementation = python.compile(
        model, config=python.Config(common_subexpression_elimination=True)
    )
    no_cse_implementation = python.compile(
        model, config=python.Config(common_subexpression_elimination=False)
    )

    # state_vector_next = python_implementation.model(0.1, state_vector)

    # random -> random_sample in 1.25
    inputs = np.random.default_rng(seed=1).random((100, 2))

    # run with cse, without cse
    print('CSE')
    cse_times = []
    for l, r in inputs:
        state_vector = np.array([[l, r]]).transpose()
        cse_start_time = datetime.now()
        cse_implementation.model(0.1, state_vector)
        cse_end_time = datetime.now()
        cse_dt = cse_end_time - cse_start_time
        cse_times.append(cse_dt)
    assert len(cse_times) > 0
    cse_p99_slowest = np.percentile(cse_times, 99)

    print('NO CSE')
    no_cse_times = []
    for l, r in inputs:
        state_vector = np.array([[l, r]]).transpose()
        no_cse_start_time = datetime.now()
        no_cse_implementation.model(0.1, state_vector)
        no_cse_end_time = datetime.now()
        no_cse_dt = no_cse_end_time - no_cse_start_time
        no_cse_times.append(no_cse_dt)
    assert len(no_cse_times) > 0
    no_cse_p01_fastest = np.percentile(no_cse_times, 1)

    # compare times
    print(f"Expect CSE {cse_p99_slowest} < No CSE {no_cse_p01_fastest}")
    assert cse_p99_slowest < no_cse_p01_fastest
