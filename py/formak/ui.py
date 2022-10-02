import numpy as np

from sympy import Symbol, symbols, simplify


class Model(object):
    def __init__(self, dt, state, control, state_model, *, debug_print=False):
        self.dt = dt
        self.state = state
        self.control = control
        self.state_model = state_model

        self.params = {}

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

    # Fit the model to data
    def fit(self, X, y=None):
        # TODO(buck): implement parameter fitting, y ignored
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
