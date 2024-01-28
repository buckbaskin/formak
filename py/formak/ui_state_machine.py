import dataclasses
import inspect
from collections import namedtuple
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from formak.exceptions import ModelFitError
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from formak import python, ui_model

SearchState = namedtuple("SearchState", ["state", "transition_path"])


class StateId(Enum):
    Start = 0
    Symbolic_Model = auto()
    Fit_Model = auto()


class ConfigView(python.Config):
    def __init__(self, params: Dict[str, Any]):
        self._params = params

        default_config = python.Config()
        for key, value in dataclasses.asdict(default_config).items():
            if key not in self._params:
                self._params[key] = value

    @property
    def common_subexpression_elimination(self) -> bool:
        return self._params["common_subexpression_elimination"]

    @property
    def python_modules(self):
        return self._params["python_modules"]

    @property
    def extra_validation(self) -> bool:
        return self._params["extra_validation"]

    @property
    def max_dt_sec(self) -> float:
        return self._params["max_dt_sec"]

    @property
    def innovation_filtering(self) -> Optional[float]:
        return self._params["innovation_filtering"]


class StateMachineState:
    def __init__(self, name: str, history: List[StateId]):
        self.name = name
        self._history = history

    @classmethod
    def state_id(cls) -> StateId:
        raise NotImplementedError()

    def history(self) -> List[StateId]:
        return self._history

    @classmethod
    def available_transitions(cls) -> List[str]:
        """
        Available function calls

        for each name in the list, getattr(state, name)(*args, **kwargs) will perform the state transition
        """
        raise NotImplementedError()

    def search(
        self, end_state: StateId, *, max_iter: int = 100, debug: bool = True
    ) -> List[str]:
        """Breadth First Search of state transitions.

        For each name in the list, next_state = getattr(current_state, name)(*args, **kwargs) will perform the next state transition
        """
        if not isinstance(end_state, StateId):
            raise ValueError(
                f"Could not match state of type {type(end_state)}, expected StateId"
            )

        frontier = [SearchState(self, [])]

        if debug:
            print("Initial State\n", frontier)

        for i in range(max_iter):
            if len(frontier) <= 0:
                break

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
            f"Could not find a path from state {self.state_id()} to desired state '{end_state}' in {i} iterations"
        )


class NisScore:
    def __call__(self, estimator: python.SklearnEKFAdapter, X, y=None) -> float:
        score = estimator.score(X=X, y=y)

        assert isinstance(score, float)

        return score


PIPELINE_STAGE_NAME = "kalman"


class FitModelState(StateMachineState):
    def __init__(
        self,
        name: str,
        history: List[StateId],
        model: ui_model.Model,
        parameter_space: Dict[str, List[Any]],
        parameter_sampling_strategy,
        data,
        cross_validation_strategy,
    ):
        super().__init__(name=name, history=history + [self.state_id()])

        ## 5 Parts of Hyper-Parameter Search problem
        # 1. Estimator
        self.symbolic_model = model

        # 2. Parameter Space
        self.parameter_space = parameter_space
        self.data = data

        required_keys = {
            "process_noise": {},
            "sensor_models": {},
            "sensor_noises": {},
            "calibration_map": {},
        }  # type: Dict[str, Dict[Any,Any]]

        for key, default in required_keys.items():
            if key not in self.parameter_space or not self.parameter_space[key]:
                self.parameter_space[key] = [default]

        # 3. Parameter Space Search/Sampling
        self.parameter_sampling_strategy = parameter_sampling_strategy

        # 4. Cross Validation
        self.cross_validation_strategy = cross_validation_strategy

        # 5. Scoring Function
        self.scoring = None

        # Last call in constructor
        self._fit_model_impl()

    @classmethod
    def state_id(cls) -> StateId:
        return StateId.Fit_Model

    @classmethod
    def available_transitions(cls) -> List[str]:
        return []

    def export_python(self) -> python.ExtendedKalmanFilter:
        return self.fit_estimator.export_python()

    def _fit_model_impl(self, debug_print=False):
        """
        This impl function contains all of the scikit-learn wrangling to
        organize it away from the logical flow of the state machine. This may
        move to its own separate helper file.
        """

        if self.cross_validation_strategy is None:
            self.cross_validation_strategy = TimeSeriesSplit

        # auto_examples/model_selection/plot_grid_search_digits.html
        # auto_examples/compose/plot_compare_reduction.html

        # NOTE: no labels here, clustering/unsupervised classification
        X = self.data

        n_samples = len(X)
        MIN_SAMPLES = 3
        if n_samples < MIN_SAMPLES:
            raise ModelFitError(
                "Model fitting requires at least %d samples for the train-test split. %d samples provided"
                % (
                    MIN_SAMPLES,
                    n_samples,
                )
            )

        X_train, X_test = train_test_split(X, test_size=0.5, random_state=1)

        adapter = python.SklearnEKFAdapter.Create(
            symbolic_model=self.symbolic_model,
            process_noise=self.parameter_space["process_noise"][0],
            sensor_models=self.parameter_space["sensor_models"][0],
            sensor_noises=self.parameter_space["sensor_noises"][0],
            calibration_map=self.parameter_space["calibration_map"][0],
            config=ConfigView({k: v[0] for k, v in self.parameter_space.items()}),
        )

        if self.scoring is None:
            self.scoring = NisScore()

        n_splits = min(len(X) - 1, 5)

        ts_cv = TimeSeriesSplit(n_splits=n_splits, gap=0)

        grid_search = GridSearchCV(
            # estimator=pipeline,
            estimator=adapter,
            param_grid=self.parameter_space,
            scoring=self.scoring,
            cv=ts_cv,
            error_score="raise",
            return_train_score=True,
        )

        grid_search.fit(X=X, y=None)

        test_scores = grid_search.cv_results_["mean_test_score"]
        estimator_params = grid_search.cv_results_["params"]
        train_scores = grid_search.cv_results_["mean_train_score"]

        if debug_print:
            for idx, (test_score, estimator_param, train_score) in enumerate(
                sorted(
                    zip(
                        test_scores,
                        estimator_params,
                        train_scores,
                    ),
                    key=lambda k: (k[0], k[2]),
                )
            ):
                print("\n--> Result", idx)
                print(test_score, train_score)
                print(
                    "innovation_filtering",
                    estimator_param["innovation_filtering"],
                )

                if idx >= 5:
                    break

        self.fit_estimator = grid_search.best_estimator_


class SymbolicModelState(StateMachineState):
    def __init__(self, name: str, history: List[StateId], model: ui_model.Model):
        super().__init__(name=name, history=history + [self.state_id()])
        self.model = model

    @classmethod
    def state_id(cls) -> StateId:
        return StateId.Symbolic_Model

    @classmethod
    def available_transitions(cls) -> List[str]:
        return ["fit_model"]

    def fit_model(
        self,
        parameter_space: Dict[str, List[Any]],
        data,
        *,
        parameter_sampling_strategy=None,
        cross_validation_strategy=None,
    ) -> FitModelState:
        """
        Symbolic Model -> Fit Model.

        Given the symbolic model contained and the given parameters for
        hyper-parameter search, perform the hyper-parameter search.
        """

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
    def state_id(cls) -> StateId:
        return StateId.Start

    @classmethod
    def available_transitions(cls) -> List[str]:
        return ["symbolic_model"]

    def symbolic_model(self, model: ui_model.Model) -> SymbolicModelState:
        return SymbolicModelState(name=self.name, history=self.history(), model=model)
