import numpy as np

from scipy.optimize import minimize
from sympy import Symbol, symbols, simplify

from formak import python


class Model(object):
    def __init__(
        self,
        dt,
        state,
        control,
        state_model,
        process_noise=None,
        sensor_models=None,
        sensor_noises=None,
        compile=False,
        *,
        debug_print=False
    ):
        self.dt = dt
        self.state = state
        self.control = control
        self.state_model = state_model

        self.params = {
            "process_noise": process_noise,
            "sensor_models": sensor_models,
            "sensor_noises": sensor_noises,
            "compile": compile,
        }

        for k in list(self.state_model.keys()):
            self.state_model[k] = simplify(self.state_model[k])

        assert len(state_model) == len(state)

        for k in state:
            try:
                assert k in state_model
            except AssertionError:
                print("%s ( %s ) missing from state model" % (k, type(k)))
                raise

        if debug_print:
            print("State Model")
            for k in sorted(list(state_model.keys()), key=lambda x: x.name):
                print("  %s: %s" % (k, state_model[k]))

    def _flatten_scoring_params(self, params):
        # "process_noise": np.eye(1),
        # "sensor_models": {"simple": {ui.Symbol("v"): ui.Symbol("v")}},
        # "sensor_noises": {"simple": np.eye(1)},
        flattened = list(np.diagonal(params["process_noise"]))
        for key in sorted(list(params["sensor_models"])):
            flattened.extend(np.diagonal(params["sensor_noises"][key]))

    def _inverse_flatten_scoring_params(self, flattened):
        params = {k: v for k, v in self.params.items()}

        controls_size = len(self.control)
        controls, flattened = flattened[:controls_size], flattened[controls_size:]

        np.fill_diagonal(params["process_noise"], controls)

        for key in sorted(list(sensor_models)):
            sensor_size = len(np.diagonal(params["sensor_noises"][key]))
            sensor, flattened = flattened[:sensor_size], flattened[sensor_size:]
            np.fill_diagonal(params["sensor_noises"][key], sensor)

    # Fit the model to data
    def fit(self, X, y=None):
        assert self.params["process_noise"] is not None
        assert self.params["sensor_models"] is not None
        assert self.params["sensor_noises"] is not None

        x0 = self._flatten_scoring_params(self.params)
        # TODO(buck): implement parameter fitting, y ignored
        def minimize_this(x):
            scoring_params = self._inverse_flatten_scoring_params(x)
            python_ekf = python.compile_ekf(
                state_model=self,
                process_noise=scoring_params["process_noise"],
                sensor_models=scoring_params["sensor_models"],
                sensor_noises=scoring_params["sensor_noises"],
                config={"compile": self.params["compile"]},
            )

        return self

    # Compute the squared Mahalanobis distances of given observations.
    def mahalanobis(self, X):
        # TODO(buck): mahalanobis
        n_samples, n_features = X.shape
        shape = (n_samples,)
        return np.zeros(shape)

    # Compute the log-likelihood of X_test under the estimated Gaussian model.
    def score(self, X):
        # TODO(buck): score
        return 0.0

    # Transform readings to innovations
    def transform(self, X):
        # TODO(buck): transform
        n_samples, n_features = X.shape
        output_features = n_features - len(self.control)

        return np.zeros((n_samples, output_features))

    # Fit the model to data and transform readings to innovations
    def fit_transform(self, X, y=None):
        # TODO(buck): Implement the combined version (return innovations calculated while fitting)
        self.fit(X, y)
        return self.transform(X)

    # Get parameters for this estimator.
    def get_params(self, deep=True) -> dict:
        return self.params

    # Set the parameters of this estimator.
    def set_params(self, **params):
        for p in params:
            if p in self.params:
                self.params[p] = params[p]

        return self
