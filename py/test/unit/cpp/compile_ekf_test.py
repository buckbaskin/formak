import pytest
from formak.exceptions import ModelConstructionError

from formak import cpp, ui


@pytest.mark.xfail(reason="Unsure on what changed with the extra_validation")
def test_EKF_model_collapse():
    config = cpp.Config()
    config.extra_validation = True

    with pytest.raises(ModelConstructionError):
        cpp.compile_ekf(
            state_model=ui.Model(
                ui.Symbol("dt"),
                set(ui.symbols(["x", "y"])),
                set(ui.symbols(["a"])),
                {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
            ),
            process_noise={ui.Symbol("a"): 1.0},
            sensor_models={},
            sensor_noises={},
            config=config,
        )
