import dataclasses
from typing import Any, Dict, Optional

from backend_py.config import Config


class ConfigView(Config):
    def __init__(self, params: Dict[str, Any]):
        self._params = params

        default_config = Config()
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
