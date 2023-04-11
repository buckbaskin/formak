import warnings
from datetime import timedelta

import numpy as np
import pytest
from hypothesis import given, reject, settings
from hypothesis.strategies import floats
from numpy.testing import assert_almost_equal

from formak import python, ui

warnings.filterwarnings("error")


@pytest.mark.skip("Marked flaky, execution time varying by 15ms to 2000ms")
@given(floats(), floats(), floats())
@settings(deadline=timedelta(seconds=3))
def test_Model_impl_property(x, y, a):
    config = {}
    dt = 0.1

    ui_Model = ui.Model(
        ui.Symbol("dt"),
        set(ui.symbols(["x", "y"])),
        set(ui.symbols(["a"])),
        {ui.Symbol("x"): ui.Symbol("x") * ui.Symbol("y"), ui.Symbol("y"): "y + a * dt"},
    )
    model = python.Model(
        ui_Model,
        config,
    )

    control_vector = np.array([[a]])
    state_vector = np.array([[x, y]]).transpose()
    if not np.isfinite(state_vector).all() or not np.isfinite(control_vector).all():
        reject()
    if (np.abs(state_vector) > 1e100).any() or (np.abs(control_vector) > 1e100).any():
        reject()

    next_state = model.model(dt=dt, state=state_vector, control_vector=control_vector)

    for i, key in enumerate(model.arglist[1 : model.state_size + 1]):
        python_version = next_state[i]

        subs_args = [
            (symbol, float(val))
            for (symbol, val) in zip(
                model.arglist, [dt] + list(state_vector) + list(control_vector)
            )
        ]
        print(subs_args)
        symbolic_version = ui_Model.state_model[key].subs(subs_args)

        assert_almost_equal(python_version, symbolic_version)
