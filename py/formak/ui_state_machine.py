import inspect
from collections import namedtuple
from typing import List

from sklearn.model_selection import TimeSeriesSplit, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from formak import python, ui_model

SearchState = namedtuple("SearchState", ["state", "transition_path"])


# TODO add testing for each of these classes, methods
# TODO add typing for each of these classes, methods


class StateMachineState:
    def __init__(self, name: str, history: List[str]):
        self.name = name
        self._history = history

    @classmethod
    def state_id(cls) -> str:
        # TODO re-evaluate the name of this function from the user perspective
        raise NotImplementedError()

    def history(self) -> List[str]:
        return self._history

    @classmethod
    def available_transitions(cls) -> List[str]:
        raise NotImplementedError()

    def search(
        self, end_state: str, *, max_iter: int = 100, debug: bool = True
    ) -> List[str]:
        """Breadth First Search of state transitions."""
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
                end_state_type = inspect.signature(
                    transition_callable
                ).return_annotation

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


class NisScore(object):
    pass


class FitModelState(StateMachineState):
    def __init__(
        self,
        name: str,
        history: List[str],
        model: ui_model.Model,
        parameter_space,
        parameter_sampling_strategy,
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

        # 3. Parameter Space Search/Sampling
        self.parameter_sampling_strategy = parameter_sampling_strategy

        # 4. Cross Validation
        self.cross_validation_strategy = cross_validation_strategy

        # 5. Scoring Function
        self.scoring = None

        # Last call in constructor
        self._fit_model_impl()

    @classmethod
    def state_id(cls) -> str:
        return "Fit Model"

    @classmethod
    def available_transitions(cls) -> List[str]:
        return []

    def _fit_model_impl(self):
        # This impl function contains all of the scikit-learn wrangling to
        # organize it away from the logical flow of the state machine. This may
        # move to its own separate helper file.

        if self.cross_validation_strategy is None:
            # TODO look up the correct scikit-learn cross validation
            self.cross_validation_strategy = TimeSeriesSplit

        if self.scoring is None:
            # TODO implement the scoring
            self.scoring = NisScore

        param_grid = {"innovation_filtering": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        # auto_examples/model_selection/plot_grid_search_digits.html
        # auto_examples/compose/plot_compare_reduction.html

        # NOTE: no labels here, clustering/unsupervised classification
        X = self.data

        X_train, X_test = train_test_split(X, test_size=0.5, random_state=1)

        model = python.compile(self.symbolic_model)

        pipeline = Pipeline([("kalman", model)])

        ts_cv = TimeSeriesSplit(n_splits=5, gap=0)

        # Maybe useful: all_splits = ts_cv.split(X)

        cv_scores = cross_validate(
            estimator=model,
            X=X,
            y=None,
            cv=ts_cv,
            scoring=self.scoring,
            error_score="raise",
            return_estimator=True,
        )

        test_score, estimator, train_score = min(
            zip(
                cv_scores["test_score"],
                cv_scores["estimator"],
                cv_scores["train_score"],
            )
        )

        self.fit_estimator = estimator


class SymbolicModelState(StateMachineState):
    def __init__(self, name: str, history: List[str], model: ui_model.Model):
        # TODO check this call syntax
        super().__init__(name=name, history=history + [self.state_id()])
        self.model = model

    @classmethod
    def state_id(cls) -> str:
        return "Symbolic Model"

    @classmethod
    def available_transitions(cls) -> List[str]:
        return ["fit_model"]

    def fit_model(
        self,
        parameter_space,
        data,
        *,
        parameter_sampling_strategy=None,
        cross_validation_strategy=None
    ) -> FitModelState:
        """
        Symbolic Model -> Fit Model.

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
        return FitModelState(
            name=self.name,
            history=self.history(),
            model=self.model,
            parameter_space=parameter_space,
            parameter_sampling_strategy=parameter_sampling_strategy,
            data=data,
            cross_validation_strategy=cross_validation_strategy,
        )


class DesignManager(StateMachineState):
    def __init__(self, name):
        super().__init__(name=name, history=[self.state_id()])

    @classmethod
    def state_id(cls) -> str:
        return "Start"

    @classmethod
    def available_transitions(cls) -> List[str]:
        return ["symbolic_model"]

    def symbolic_model(self, model: ui_model.Model) -> SymbolicModelState:
        return SymbolicModelState(name=self.name, history=self.history(), model=model)
