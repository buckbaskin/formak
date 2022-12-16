import numpy as np
import warnings

from datetime import datetime, timedelta
from scipy.stats import multivariate_normal
from hypothesis import given, reject, settings
from hypothesis.strategies import floats
from numpy.testing import assert_almost_equal

from formak import ui, python

warnings.filterwarnings("error")


def test_EKF_process_with_control():
    config = python.Config()
    dt = 0.1

    ekf = python.ExtendedKalmanFilter(
        state_model=ui.Model(
            ui.Symbol("dt"),
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
        ),
        process_noise=np.eye(1),
        sensor_models={},
        sensor_noises={},
        config=config,
    )

    control_vector = np.array([[0.2]])
    covariance = np.eye(2)

    state_vector = np.array([[0.0, 0.0]]).transpose()
    assert_almost_equal(
        ekf.process_model(
            dt=dt, state=state_vector, covariance=covariance, control=control_vector
        )[0].transpose(),
        [[0.0, 0.02]],
    )

    state_vector = np.array([[0.0, 1.0]]).transpose()
    assert_almost_equal(
        ekf.process_model(
            dt=dt, state=state_vector, covariance=covariance, control=control_vector
        )[0].transpose(),
        [[0.0, 1.02]],
    )

    state_vector = np.array([[1.0, 0.0]]).transpose()
    assert_almost_equal(
        ekf.process_model(
            dt=dt, state=state_vector, covariance=covariance, control=control_vector
        )[0].transpose(),
        [[0.0, 0.02]],
    )

    state_vector = np.array([[1.0, 1.0]]).transpose()
    assert_almost_equal(
        ekf.process_model(
            dt=dt, state=state_vector, covariance=covariance, control=control_vector
        )[0].transpose(),
        [[1.0, 1.02]],
    )


def test_EKF_sensor():
    config = python.Config()

    ekf = python.ExtendedKalmanFilter(
        state_model=ui.Model(
            ui.Symbol("dt"),
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
        ),
        process_noise=np.eye(1),
        sensor_models={
            "simple": {"reading1": ui.Symbol("x")},
            "combined": {"reading2": ui.Symbol("x") + ui.Symbol("y")},
        },
        sensor_noises={"simple": np.eye(1), "combined": np.eye(1)},
        config=config,
    )

    covariance = np.eye(2)
    reading = 1.0
    state_vector = np.array([[0.0, 0.0]]).transpose()

    next_state, next_cov = ekf.sensor_model(
        "simple",
        state=state_vector,
        covariance=covariance,
        sensor_reading=np.array([[reading]]),
    )
    assert abs(reading - next_state[0]) < abs(reading - state_vector[0])

    next_state, next_cov = ekf.sensor_model(
        "combined",
        state=state_vector,
        covariance=covariance,
        sensor_reading=np.array([[reading]]),
    )
    assert abs(reading - next_state[0]) < abs(reading - state_vector[0])
    assert abs(reading - next_state[1]) < abs(reading - state_vector[1])


def test_EKF_process_jacobian():
    config = python.Config()
    dt = 0.1

    ekf = python.ExtendedKalmanFilter(
        state_model=ui.Model(
            ui.Symbol("dt"),
            set(ui.symbols(["x", "y"])),
            set(ui.symbols(["a"])),
            {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
        ),
        process_noise=np.eye(1),
        sensor_models={},
        sensor_noises={},
        config=config,
    )

    control_vector = np.array([[0.2]])

    state_vector = np.array([[0.0, 0.0]]).transpose()
    assert_almost_equal(
        ekf.process_jacobian(dt=dt, state=state_vector, control=control_vector),
        [[0.0, 0.0], [0.0, 1.0]],
    )

    state_vector = np.array([[0.0, 1.0]]).transpose()
    assert_almost_equal(
        ekf.process_jacobian(dt=dt, state=state_vector, control=control_vector),
        [[1.0, 0.0], [0.0, 1.0]],
    )

    state_vector = np.array([[1.0, 0.0]]).transpose()
    assert_almost_equal(
        ekf.process_jacobian(dt=dt, state=state_vector, control=control_vector),
        [[0.0, 1.0], [0.0, 1.0]],
    )

    state_vector = np.array([[1.0, 1.0]]).transpose()
    assert_almost_equal(
        ekf.process_jacobian(dt=dt, state=state_vector, control=control_vector),
        [[1.0, 1.0], [0.0, 1.0]],
    )


@given(floats(), floats(), floats())
@settings(deadline=timedelta(seconds=2))
def test_EKF_process_property(state_x, state_y, control_a):
    config = {}
    dt = 0.1

    ui_Model = ui.Model(
        ui.Symbol("dt"),
        set(ui.symbols(["x", "y"])),
        set(ui.symbols(["a"])),
        {
            ui.Symbol("x"): "x + y * dt",
            ui.Symbol("y"): "y + a * dt",
        },
    )
    ekf = python.compile_ekf(
        state_model=ui_Model,
        process_noise=np.eye(1),
        sensor_models={},
        sensor_noises={},
        config=config,
    )

    control_vector = np.array([[control_a]])
    covariance = np.eye(2)
    state_vector = np.array([[state_x, state_y]]).transpose()

    if not np.isfinite(state_vector).all() or not np.isfinite(control_vector).all():
        # reject infinite / NaN inputs
        reject()
    if (np.abs(state_vector) > 1e100).any() or (np.abs(control_vector) > 1e100).any():
        # reject poorly sized inputs
        reject()

    next_state, next_cov = ekf.process_model(
        dt=dt, state=state_vector, covariance=covariance, control=control_vector
    )

    for i, key in enumerate(ekf.state_model.arglist[1 : ekf.state_size + 1]):
        python_version = next_state[i]

        subs_args = [
            (symbol, float(val))
            for (symbol, val) in zip(
                ekf.state_model.arglist,
                [dt] + list(state_vector) + list(control_vector),
            )
        ]
        symbolic_version = ui_Model.state_model[key].subs(subs_args)

        assert_almost_equal(python_version, symbolic_version)

    starting_central_probability = multivariate_normal(cov=covariance).pdf(
        np.zeros_like(state_vector).transpose()
    )
    try:
        ending_central_probability = multivariate_normal(cov=next_cov).pdf(
            np.zeros_like(state_vector).transpose()
        )
    except np.linalg.LinAlgError:
        # reject inputs leading to poorly conditions covariances
        reject()

    try:
        assert ending_central_probability < starting_central_probability
    except AssertionError:
        print("Starting at %f" % (starting_central_probability,))
        print(covariance)
        print("Ending at %f" % (ending_central_probability,))
        print(next_cov)
        raise


@given(floats(), floats(), floats())
def test_EKF_sensor_property(x, y, a):
    config = {}

    ui_Model = ui.Model(
        ui.Symbol("dt"),
        set(ui.symbols(["x", "y"])),
        set(ui.symbols(["a"])),
        {
            ui.Symbol("x"): ui.Symbol("x") * ui.Symbol("y") + ui.Symbol("x"),
            ui.Symbol("y"): "y + a * dt",
        },
    )
    ekf = python.compile_ekf(
        state_model=ui_Model,
        process_noise=np.eye(1),
        sensor_models={"simple": {ui.Symbol("x"): ui.Symbol("x")}},
        sensor_noises={"simple": np.eye(1)},
        config=config,
    )

    control_vector = np.array([[a]])
    covariance = np.eye(2)
    state_vector = np.array([[x, y]]).transpose()

    if not np.isfinite(state_vector).all() or not np.isfinite(control_vector).all():
        # reject infinite / NaN inputs
        reject()
    if (np.abs(state_vector) > 1e100).any() or (np.abs(control_vector) > 1e100).any():
        # reject poorly sized inputs
        reject()

    next_state, next_cov = ekf.sensor_model(
        "simple", state=state_vector, covariance=covariance, sensor_reading=np.eye(1)
    )
    first_innovation = ekf.innovations["simple"]

    try:
        starting_central_probability = multivariate_normal(cov=covariance).pdf(
            np.zeros_like(state_vector).transpose()
        )
        ending_central_probability = multivariate_normal(cov=next_cov).pdf(
            np.zeros_like(state_vector).transpose()
        )
    except np.linalg.LinAlgError:
        print("starting cov")
        print(covariance)
        print("next_cov")
        print(next_cov)
        raise

    try:
        # more confident in state estimate after sensor update
        assert ending_central_probability > starting_central_probability
    except AssertionError:
        print("starting pdf")
        print(starting_central_probability)
        print("ending pdf")
        print(ending_central_probability)
        raise

    ekf.sensor_model(
        "simple", state=next_state, covariance=next_cov, sensor_reading=np.eye(1)
    )
    second_innovation = ekf.innovations["simple"]

    # state moves towards reading after a sensor update
    assert np.linalg.norm(first_innovation) > np.linalg.norm(second_innovation)


def test_EKF_sensor_property_failing_example():
    start_time = datetime.now()
    x, y, a = (-538778789133922.0, -538778789133922.0, -2.6221616798653463e-203)
    config = {}

    ui_Model = ui.Model(
        ui.Symbol("dt"),
        set(ui.symbols(["x", "y"])),
        set(ui.symbols(["a"])),
        {
            ui.Symbol("x"): ui.Symbol("x") * ui.Symbol("y") + ui.Symbol("x"),
            ui.Symbol("y"): "y + a * dt",
        },
    )
    ekf = python.compile_ekf(
        state_model=ui_Model,
        process_noise=np.eye(1),
        sensor_models={"simple": {ui.Symbol("x"): ui.Symbol("x")}},
        sensor_noises={"simple": np.eye(1)},
        config=config,
    )

    control_vector = np.array([[a]])
    covariance = np.eye(2)
    state_vector = np.array([[x, y]]).transpose()

    if not np.isfinite(state_vector).all() or not np.isfinite(control_vector).all():
        # reject infinite inputs
        reject()
    if (np.abs(state_vector) > 1e100).any() or (np.abs(control_vector) > 1e100).any():
        # reject poorly sized inputs
        reject()

    next_state, next_cov = ekf.sensor_model(
        "simple", state=state_vector, covariance=covariance, sensor_reading=np.eye(1)
    )
    first_innovation = ekf.innovations["simple"]

    try:
        starting_central_probability = multivariate_normal(cov=covariance).pdf(
            np.zeros_like(state_vector).transpose()
        )
        ending_central_probability = multivariate_normal(cov=next_cov).pdf(
            np.zeros_like(state_vector).transpose()
        )
    except np.linalg.LinAlgError:
        print("starting cov")
        print(covariance)
        print("next_cov")
        print(next_cov)
        raise

    try:
        # more confident in state estimate after sensor update
        assert ending_central_probability > starting_central_probability
    except AssertionError:
        print("starting pdf")
        print(starting_central_probability)
        print("ending pdf")
        print(ending_central_probability)
        raise

    ekf.sensor_model(
        "simple", state=next_state, covariance=next_cov, sensor_reading=np.eye(1)
    )
    second_innovation = ekf.innovations["simple"]

    # state moves towards reading after a sensor update
    assert np.linalg.norm(first_innovation) > np.linalg.norm(second_innovation)

    end_time = datetime.now()

    elapsed = end_time - start_time

    assert elapsed < timedelta(seconds=0.1)
