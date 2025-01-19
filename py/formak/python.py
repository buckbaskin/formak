from __future__ import annotations

import dataclasses
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

import sympy
from backend_py.compile_ekf import compile_ekf
from backend_py.config import Config
from backend_py.extended_kalman_filter import (
    ExtendedKalmanFilter,
    assert_valid_covariance,
    nearest_positive_definite,
)
from backend_py.named_vector import named_vector
from backend_py.sensor_model import SensorModel
from formak.exceptions import MinimizationFailure, ModelConstructionError
from problemdefinition.ui_model_base import UiModelBase
from sympy import Symbol
