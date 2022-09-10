from sympy.utilities.lambdify import lambdify
from numba import njit

MODULES = ["scipy", "numpy", "math"]


class Model(object):
    """Python implementation of the model"""

    def __init__(self, symbolic_model, config):
        # assert isinstance(symbolic_model, ui.Model)

        self.arglist = (
            [symbolic_model.dt]
            + sorted(list(symbolic_model.state), key=lambda x: x.name)
            + sorted(list(symbolic_model.control), key=lambda x: x.name)
        )

        # TODO(buck): Common Subexpression Elimination supports multiple inputs, so use common subexpression elimination across state calculations
        self._impl = [
            lambdify(
                self.arglist,
                a,
                modules=MODULES,
                cse=config.common_subexpression_elimination,
            )
            for a in self.arglist[: len(symbolic_model.state)]
        ]

        if config.compile:
            self._impl = [njit(i) for i in self._impl]

    def _model(self, dt, state, control):
        assert isinstance(dt, float)
        assert isinstance(state, list)
        assert isinstance(control, list)

        for impl in self._impl:
            yield impl(*([dt] + state + control))

    # TODO(buck): numpy -> numpy if not compiled
    #   - Given the arglist, refactor expressions to work with state vectors
    #   - Transparently convert states, controls to numpy so that it's always numpy -> numpy
    def model(self, dt, state, control=None):
        if control is None:
            control = []
        return list(self._model(dt, state, control))


class ExtendedKalmanFilter(object):
    def __init__(self, state_model, sensor_models, config):
        self.state_model = state_model
        self.sensor_models = sensor_models

        self.arglist_sensor = sorted(
            list(state_model.state), key=lambda x: x.name
        ) + sorted(list(state_model.control), key=lambda x: x.name)
        self.arglist_process = [state_model.dt] + self.arglist_sensor

        self._process_impl = [
            lambdify(
                self.arglist_process,
                a,
                modules=MODULES,
                cse=config.common_subexpression_elimination,
            )
            for a in self.arglist_process[: len(state_model.state)]
        ]

    def _process_model(self, dt, state, control):
        assert isinstance(dt, float)
        assert isinstance(state, list)
        assert isinstance(control, list)

        for impl in self._process_impl:
            yield impl(*([dt] + state + control))

    # TODO(buck): numpy -> numpy if not compiled
    def process_model(self, dt, state, covariance, control):
        # TODO(buck): Implement EKF process update variance
        return list(self._process_model(dt, state, control))

    def sensor_model(self, state, covariance, sensor):
        # TODO(buck): Implement EKF sensor update
        return state, covariance


class Config(object):
    def __init__(self, compile=False, common_subexpression_elimination=True):
        self.compile = compile
        self.common_subexpression_elimination = common_subexpression_elimination


def compile(symbolic_model, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    return Model(symbolic_model, config)


def compile_ekf(state_model, sensor_models, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    return ExtendedKalmanFilter(state_model, sensor_models, config)
