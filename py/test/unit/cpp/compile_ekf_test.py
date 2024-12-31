import pytest
from formak.exceptions import ModelConstructionError
from formak.problemdefinition.model import Model as UiModel
from sympy import Symbol, symbols

from formak import cpp


@pytest.mark.xfail(reason="Unsure on what changed with the extra_validation")
def test_EKF_model_collapse():
    config = cpp.Config()
    config.extra_validation = True

    with pytest.raises(ModelConstructionError):
        cpp.compile_ekf(
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
