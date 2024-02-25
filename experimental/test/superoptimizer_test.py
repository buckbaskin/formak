from superoptimizer import superoptimizer, astar, StateBase, ActionBase
from collections import namedtuple

# A -> B
# |      \
# C ----> D

GraphEdge = namedtuple(
    "GraphEdge",
    [
        "to_node",
        "distance",
    ],
)

graph = {
    "A": [
        GraphEdge("B", 1),
        GraphEdge(
            "C",
            1,
        ),
    ],
    "B": [
        GraphEdge("D", 1),
    ],
    "C": [
        GraphEdge("D", 3),
    ],
    "D": [],
}

heuristic_map = {
    "A": 0.0,
    "B": 0.5,
    "C": 0.0,
    "D": 0.0,
}


class State(StateBase):
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"State({self._name})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return self._name == other._name

    def __hash__(self):
        return hash(self._name)


class Action(ActionBase):
    def __init__(self, from_: State, to: State, cost: float):
        assert isinstance(from_, State)
        assert isinstance(to, State)
        assert isinstance(cost, (float, int))

        self.from_ = from_
        self.to = to
        self.cost = cost

    def __eq__(self, other: ActionBase):
        if not isinstance(other, type(self)):
            return False

        return (self.from_, self.to, self.cost) == (
            other.from_,
            other.to,
            other.cost,
        )

    def __repr__(self):
        return f"Action({self.from_}, {self.to}, {self.cost})"


def heuristic(state):
    return heuristic_map[state._name]


def available_edges(state):
    return [Action(state, State(to), cost) for to, cost in graph[state._name]]


def transition(state, action):
    assert action.from_ == state
    return action.to


def test_astar_one():
    result = astar(
        initial_state=State("C"),
        goal_state=State("D"),
        transition=transition,
        heuristic=heuristic,
        edges=available_edges,
    )
    assert result == [
        Action(State("C"), State("D"), 3),
    ]


def test_astar_end():
    result = astar(
        initial_state=State("D"),
        goal_state=State("D"),
        transition=transition,
        heuristic=heuristic,
        edges=available_edges,
    )
    assert result == []


def test_astar_full():
    result = astar(
        initial_state=State("A"),
        goal_state=State("D"),
        transition=transition,
        heuristic=heuristic,
        edges=available_edges,
    )
    assert result == [
        Action(State("A"), State("B"), 1.0),
        Action(State("B"), State("D"), 1.0),
    ]


def test_superoptimizer():
    result = superoptimizer()
    assert result == ["add", "mul"]
