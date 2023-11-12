from collections import namedtuple

SearchState = namedtuple("SearchState", ["state", "transition_path"])


# TODO add testing for each of these classes, methods
# TODO add typing for each of these classes, methods

class StateMachineState(object):
    def __init__(self, name: str, history: List[str]):
        self.name = name
        self._history = history

    def state_id(self) -> str:
        # TODO re-evaluate the name of this function from the user perspective
        raise NotImplementedError()

    def history(self) -> List[str]:
        return _history

    def available_transitions(self) -> List[str]:
        raise NotImplementedError()

    def search(
        self, end_state: str, *, max_iter: int = 100, debug: bool = True
    ) -> List[str]:
        """
        Breadth First Search of state transitions
        """
        # TODO this could be a queue / deque
        frontier = [SearchState(self, [])]

        if debug:
            print("Initial State\n", frontier)

        for i in range(max_iter):
            current_state, transitions = frontier[0]
            frontier = frontier[1:]

            if current_state.state_id() == end_state:
                return transitions

            for transition_name in current_state.available_transitions():
                transition_callable = getattr(current_state, transition_name)
                # TODO fix this inspect module/class usage
                end_state_type = inspect.Inspect(transition_callable).return_type

                frontier.append(
                    SearchState(end_state_type, transitions + [transition_name])
                )

                if debug:
                    print(i, "Adding", frontier[-1])

            if debug:
                print("State After", i, "\n", frontier)
        raise ValueError(
            "Could not find a path from state {self.state_id()} to {end_state} in {max_iter} iterations"
        )


class DesignManager(StateMachineState):
    def __init__(self, name):
        super().__init__(name=name, history=[self.state_id()])

    def state_id(self) -> str:
        return "Start"

    def available_transitions(self) -> List[str]:
        return ["symbolic_model"]

    def symbolic_model(self, model: ui.Model) -> SymbolicModelState:
        return SymbolicModelState(name=self.name, history=self.history(), model=model)
