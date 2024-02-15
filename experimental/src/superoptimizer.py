class ActionBase():
    def cost(self) -> float:
        raise NotImplementedError()

class StateBase():
    def __init__(self, cost_to_reach:float, path: list[ActionBase]):
        self.cost_to_reach = cost_to_reach
        self.path = path

    def alternate_path(self, cost_to_reach:float, path: list[ActionBase]) -> None:
        if cost_to_reach < self.cost_to_reach:
            self.cost_to_reach = cost_to_reach
            self.path = path

    def possible_cost(self, goal) -> float:
        return self.cost_to_reach + self.heuristic(goal)

    def heuristic(self, goal: StateBase) -> float:
        raise NotImplementedError()

    def available_edges(self) -> list[ActionBase]:
        raise NotImplementedError()

    def next_state(self, action: ActionBase) -> StateBase:
        raise NotImplementedError()

class Frontier():
    def __init__(self):
        self.q = []

    def extend(self, state: StateBase, possible_actions: list[StateBase]):
        for action in possible_actions:
            possible_state = state.next_state(action)

            try:
                key = self.q.index(possible_state)

                self.q[key].alternate_path(state.cost_to_reach + action.cost(), state.path + [action])
            except ValueError:
                # state not in frontier
                self.q.append(possible_state)

        sort(self.q, key=lambda state: state.possible_cost())
        print([state.possible_cost() for state in self.q])
        1/0

    def pop(self) -> StateBase:
        if len(self.q) == 0:
            raise ValueError('Empty Frontier')

        result = self.q[0]
        self.q = self.q[1:]

        return result


def astar(initial_state: StateBase, goal: StateBase, *, max_iterations):
    visited_states = set([initial_state])
    frontier = initial_state.available_edges()

    for i in range(max_iterations):
        if len(frontier) == 0:
            break

        next_state = frontier[0]
        frontier = frontier[1:]

        if next_state == goal:
            # done
            raise AttributeError('done, but do not know how I got here')

        frontier.extend(next_state, next_state.available_edges())
        
        pass

    raise ValueError('Failed to terminate in max iterations')


def superoptimizer():
    pass
