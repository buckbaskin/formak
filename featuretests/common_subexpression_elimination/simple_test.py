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

    leaves_count = 32

    leaves = [l, r] * leaves_count

    nodes = list(combine_nodes(leaves))

    while len(nodes) > 1:
        nodes = list(combine_nodes(nodes))

    symbolic_model = nodes[0]

    # do cse

    # run with cse, without cse

    # compare times
    1 / 0
