import pytest

from backend_cpp.compile_model import compile_model
from backend_cpp.config import Config
from formak.exceptions import ModelConstructionError
from problemdefinition.model import Model as UiModel
from sympy import Symbol, symbols


@pytest.mark.xfail(reason="Unsure on what changed with the extra_validation")
def test_EKF_model_collapse():
    config = Config()
    config.extra_validation = True

    with pytest.raises(ModelConstructionError):
        compile_model(
            state_model=UiModel(
                Symbol("dt"),
                set(symbols(["x", "y"])),
                set(symbols(["a"])),
                {Symbol("x"): "x * y", Symbol("y"): "y + a * dt"},
            ),
            process_noise={Symbol("a"): 1.0},
            sensor_models={},
            sensor_noises={},
            config=config,
        )
