import sys

from formak.ui.model import Model as UiModel
from sympy import Symbol, cos, sin, symbols

# # UI Model dependency not in the path
# print("model_builder path")
# for line in sorted(list(sys.path)):
#     print(line)
# 1 / 0


def combine_nodes(leaves):
    for i in range(0, len(leaves) // 2):
        left = leaves[i * 2]
        right = leaves[i * 2 + 1]
        yield 2 * sin(left * right) * cos(left + right)


def ui_model():
    """
    Recursively combine equivalent left and right sub-structures to create a structure with maximal amounts of common subexpressions.

    Every sub-expression has at least one common subexpression and the left and
    right states in the model share a common expression.
    """
    left, right = symbols(["left", "right"])

    leaves_count = 1024

    leaves = [left, right] * leaves_count

    nodes = list(combine_nodes(leaves))

    while len(nodes) > 1:
        nodes = list(combine_nodes(nodes))

    symbolic_model = nodes[0]

    dt = Symbol("dt")

    state = {left, right}
    state_model = {
        left: symbolic_model,
        right: symbolic_model,
    }
    control = {}

    model = UiModel(dt=dt, state=state, control=control, state_model=state_model)

    return model
