from runtime_py.sklearn import SklearnEKFAdapter


class NisScore:
    def __call__(self, estimator: SklearnEKFAdapter, X, y=None) -> float:
        score = estimator.score(X=X, y=y)

        assert isinstance(score, float)

        return score
