from superoptimizer import superoptimizer, astar
from collections import namedtuple

# A -> B
# |      \
# C ----> D

GraphEdge = namedtuple('GraphEdge', ['to_node', 'distance',])

graph = {
    'A': [GraphEdge('B', 1), GraphEdge('C', 1, ),],
    'B': [GraphEdge('D', 1),],
    'C': [GraphEdge('D', 3),],
    'D': [],
        }

heuristic_map = {
        'A': 0.0,
        'B': 0.5,
        'C': 0.0,
        'D': 0.0,
        }


def test_astar():
    class State(StateBase):
        def __init__(self, name):
            self.name = name

    class Action(ActionBase):
        def __init__(self, from_, to):
            self.from_ = from_
            self.to = to

    def heuristic(state):
        return heuristic_map[state.name]

    def cost(action):
        for edge in graph[action.from_]:
            if edge.to_node == action.to:
                return edge.distance

        raise ValueError('No matching edge in graph')

    def available_edges(state):
        return graph[state.name]

    result = astar(initial_state = State('A'), heuristic=heuristic, cost=cost, available_edges=available_edges)
    assert result == ["a", "b"]


def test_superoptimizer():
    result = superoptimizer()
    assert result == ["add", "mul"]
