import numpy as np
from dataclasses import dataclass
from typing import Any

DEFAULT_MODULES = ("scipy", "numpy", "math", {"sec": lambda v: 1.0 / np.cos(v)})


@dataclass(frozen=True)
class Config:
    """
    Options for generating C++.

    common_subexpression_elimination:
        Remove common shared computation
    python_modules:
        Allow dependencies. Math is the Python standard library
    extra_validation:
        Catch errors earlier in exchange for increased compute time
    """

    common_subexpression_elimination: bool = True
    python_modules: tuple[Any, Any, Any, Any] = DEFAULT_MODULES
    extra_validation: bool = False
    max_dt_sec: float = 0.1
    innovation_filtering: float | None = 5.0
