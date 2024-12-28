from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def assert_valid_covariance(
    covariance: NDArray, *, name: str = "Covariance", negative_tol: float = -1e-15
):
    """
    Check that the covariance array is well formed:

    - symmetric (approximately)
    - positive semidefinite (approximately)
    """
    assert isinstance(covariance, np.ndarray)
    assert np.allclose(covariance, covariance.T)

    covariance_eigenvalues = np.linalg.eig(covariance)[0]
    if np.any(covariance_eigenvalues < negative_tol):
        # negative definite matrix is not a valid representation of uncertainty
        raise AssertionError(
            f"Negative {str(name)}:\n{covariance}\nEigen Values: {min(covariance_eigenvalues)}\n{covariance_eigenvalues}"
        )
