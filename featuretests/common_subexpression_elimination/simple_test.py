from datetime import datetime
from functools import partial

import numpy as np
import pytest
from formak.microbenchmark import microbenchmark
from formak.ui import *
from sympy import cos, sin, symbols

from common import ui_model
from formak import python


def combine_nodes(leaves):
    for i in range(0, len(leaves) // 2):
        l = leaves[i * 2]
        r = leaves[i * 2 + 1]
        yield 2 * sin(l * r) * cos(l + r)


def test_python_CSE():
    model = ui_model()

    cse_implementation = python.compile(
        model, config=python.Config(common_subexpression_elimination=True)
    )
    no_cse_implementation = python.compile(
        model, config=python.Config(common_subexpression_elimination=False)
    )

    # random -> random_sample in 1.25
    inputs = np.random.default_rng(seed=1).random((101, 2))
    inputs = [np.array([[l, r]]).transpose() for l, r in inputs]

    # run with CSE, without CSE
    print("CSE")
    cse_times = microbenchmark(partial(cse_implementation.model, 0.1), inputs)
    cse_p99_slowest = np.percentile(cse_times, 99)
    print("micro cse", cse_p99_slowest)

    print("NO CSE")
    no_cse_times = microbenchmark(partial(no_cse_implementation.model, 0.1), inputs)
    no_cse_p01_fastest = np.percentile(no_cse_times, 1)
    print("micro no cse", no_cse_p01_fastest)

    # compare times
    print(f"Expect CSE {cse_p99_slowest} < No CSE {no_cse_p01_fastest}")
    assert cse_p99_slowest < no_cse_p01_fastest
