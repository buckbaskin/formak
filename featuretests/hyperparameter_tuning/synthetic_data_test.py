"""
# Hyper-Parameter Tuning Feature Test

Demonstrate tuning a model for two different innovation filtering hyper-parameters using the same process.
"""

from collections import namedtuple

from formak.ui import DManager
from model import symbolic_model

from data import generate_data
from formak import ui

SearchState = namedtuple("SearchState", ["state", "transition_path"])


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


class FitModelState(StateMachineState):
    def __init__(
        self,
        name: str,
        history: List[str],
        model: ui.Model,
        parameter_space,
        data,
        cross_validation_strategy,
    ):
        # TODO check this call syntax
        super().__init__(name=name, history=history + [self.state_id()])

        ## 5 Parts of Hyper-Parameter Search problem
        # 1. Estimator
        self.symbolic_model = model

        # 2. Parameter Space
        self.parameter_space = parameter_space
        self.data = data

        # 3. Cross Validation
        self.cross_validation_strategy = cross_validation_strategy

        # 4. ? TODO figure this out

        # 5. Scoring Function
        self.score = None

        # Last call in constructor
        self._fit_model_impl()

    def state_id(self) -> str:
        return "Fit Model"

    def available_transitions(self) -> List[str]:
        return []

    def _fit_model_impl(self):
        # This impl function contains all of the scikit-learn wrangling to
        # organize it away from the logical flow of the state machine. This may
        # move to its own separate helper file.

        if self.cross_validation_strategy is None:
            # TODO look up the correct scikit-learn cross validation
            self.cross_validation_strategy = Timeseries

        if self.scoring is None:
            # TODO implement the scoring
            self.scoring = NIS

        1 / 0


class SymbolicModelState(StateMachineState):
    def __init__(self, name: str, history: List[str], model: ui.Model):
        # TODO check this call syntax
        super().__init__(name=name, history=history + [self.state_id()])

    def state_id(self) -> str:
        return "Symbolic Model"

    def available_transitions(self) -> List[str]:
        return ["fit_model"]

    def fit_model(
        self, parameter_space, data, *, cross_validation_strategy=None
    ) -> FitModelState:
        """
        Symbolic Model -> Fit Model

        Given the symbolic model contained and the given parameters for
        hyper-parameter search, perform the hyper-parameter search.
        """
        # TODO parameter space should at least be heavily inspired by
        # scikit-learn parameter space

        # TODO parameter space may need to have fixed/default params and then
        # vary the rest?

        # TODO parameter space needs a type

        # TODO data needs a type (sklearn database? dataset? you know the one
        # I'm talking about)

        # TODO have an optional "release" config flag that will be set to true
        # and when set will apply simplify, etc aggressively.  When "release"
        # is False, optimize for fast iteration vs fastest model
        1 / 0


# TODO fix this class name
class DManager(StateMachineState):
    def __init__(self, name):
        super().__init__(name=name, history=[self.state_id()])

    def state_id(self) -> str:
        return "Start"

    def available_transitions(self) -> List[str]:
        return ["symbolic_model"]

    def symbolic_model(self, model: ui.Model) -> SymbolicModelState:
        return SymbolicModelState(name=self.name, history=self.history(), model=model)


def test_with_synthetic_data():
    true_innovation = 5

    initial_state = DManager(name="mercury")

    # Q: No-discard but for python?
    symbolic_model_state = initial_state.symbolic_model(model=symbolic_model)

    fit_model_state = symbolic_model_state.fit_model(
        parameter_space={}, data=generate_data(true_innovation)
    )

    # Note: not a state transition
    python_model = fit_model_state.export_python()

    assert (
        true_innovation - 0.5
        < python_model.config.innovation_filtering
        < true_innovation + 0.5
    )


def test_state_machine_interface():
    initial_state = DManager(name="mercury")

    # TODO: make the state names enums
    assert symbolic_model_state.history() == ["Start"]
    assert symbolic_model_state.available_transitions() == ["symbolic_model"]
    assert symbolic_model_state.search("Fit Model") == ["symbolic_model", "fit_model"]

    # Q: No-discard but for python?
    symbolic_model_state = initial_state.symbolic_model(model=symbolic_model)

    assert symbolic_model_state.history() == ["Start", "Symbolic Model"]
    assert symbolic_model_state.available_transitions() == ["fit_model"]
    assert symbolic_model_state.search("Fit Model") == ["fit_model"]
