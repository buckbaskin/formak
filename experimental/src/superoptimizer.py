from __future__ import annotations
from collections import namedtuple


class ActionBase:
    def cost(self) -> float:
        raise NotImplementedError()


class StateBase:
    def __eq__(self, other: StateBase) -> bool:
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()


FrontTuple = namedtuple("FrontTuple", ["state", "path", "reach", "heuristic"])


class Frontier:
    def __init__(self, initial_state):
        self.q = [FrontTuple(initial_state, [], 0.0, 0.0)]

    def append(
        self,
        state: StateBase,
        path: List[ActionBase],
        reach: float,
        heuristic: float,
        *,
        debug=False,
    ):
        self.q.append(FrontTuple(state, path, reach, heuristic))

        self.q.sort(key=lambda t: t.reach + t.heuristic)
        if debug:
            print(
                [
                    (t.state._name, t.reach, t.heuristic, t.reach + t.heuristic)
                    for t in self.q
                ]
            )

    def pop(self) -> StateBase:
        if len(self.q) == 0:
            raise ValueError("Empty Frontier")

        result = self.q[0]
        self.q = self.q[1:]

        return result


def astar(
    initial_state: StateBase,
    goal_state: StateBase,
    transition,
    heuristic,
    edges,
    *,
    max_iter=1000,
):
    frontier = Frontier(initial_state)
    visited = set()

    for i in range(max_iter):
        state, path, reach_cost, _ = frontier.pop()

        if state == goal_state:
            return path

        if state in visited:
            continue

        visited.add(state)

        for action in edges(state):
            next_state = transition(state, action)
            frontier.append(
                next_state,
                path + [action],
                reach_cost + action.cost,
                heuristic(next_state),
            )

    raise ValueError("Failed to terminate in max iterations")


def superoptimizer():
    pass
