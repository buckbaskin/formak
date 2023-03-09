#include <formak/testing/stats.h>
#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>
#include <unit/simple-ekf.h>  // Generated

namespace integration {

RC_GTEST_PROP(CppModel, EKF_process_property, (double x, double y, double a)) {
  // def test_EKF_process_property(state_x, state_y, control_a):
  unit::ExtendedKalmanFilter ekf;
  double dt = 0.1;

  unit::Control control({a});
  unit::State state({x, y});
  unit::Covariance covariance;

  RC_PRE(std::isfinite(x) && std::isfinite(y) && std::isfinite(a));
  RC_PRE(std::abs(x) < 1e100 && std::abs(y) < 1e100 && std::abs(a) < 1e100);

  auto next = ekf.process_model(dt, {state, covariance}, control);

  RC_ASSERT(next.state.x() == x * y);
  RC_ASSERT(next.state.y() == y + a * dt);

  FAIL() << "Need to check to see if the example I'm building from has more "
            "test assertions";
}

RC_GTEST_PROP(CppModel, EKF_sensor_property, (double x, double y)) {
  using formak::testing::stats::MultivariateNormal;

  // def test_EKF_sensor_property(x, y, a):
  unit::ExtendedKalmanFilter ekf;

  unit::State state({x, y});
  unit::Covariance covariance;

  RC_PRE(std::isfinite(x) && std::isfinite(y));
  RC_PRE(std::abs(x) < 1e100 && std::abs(y) < 1e100);

  double reading = 1.0;
  unit::SensorReading<(unit::SensorId::SIMPLE), unit::Simple> simple_reading{
      unit::Simple({reading})};
  auto next = ekf.sensor_model({state, covariance}, simple_reading);

  auto maybe_first_innovation = ekf.innovations(unit::SensorId::SIMPLE);
  RC_ASSERT(maybe_first_innovation.has_value());
  typename unit::State::DataT first_innovation = maybe_first_innovation.value();

  // try
  double starting_central_probability =
      MultivariateNormal(unit::State{}, covariance).pdf(unit::State{});
  double ending_central_probability =
      MultivariateNormal(unit::State{}, next.covariance).pdf(unit::State{});
  //     except np.linalg.LinAlgError:

  RC_ASSERT(ending_central_probability < starting_central_probability);

  // Run this to get a second innovation
  ekf.sensor_model(next, simple_reading);

  auto maybe_second_innovation = ekf.innovations(unit::SensorId::SIMPLE);
  RC_ASSERT(maybe_second_innovation.has_value());
  typename unit::State::DataT second_innovation =
      maybe_second_innovation.value();

  RC_ASSERT(first_innovation.norm() > second_innovation.norm() ||
            first_innovation.norm() == 0.0);

  // def test_EKF_sensor_property(x, y, a):
  //     config = {}
  //
  //     ui_Model = ui.Model(
  //         ui.Symbol("dt"),
  //         set(ui.symbols(["x", "y"])),
  //         set(ui.symbols(["a"])),
  //         {
  //             ui.Symbol("x"): ui.Symbol("x") * ui.Symbol("y") +
  //             ui.Symbol("x"), ui.Symbol("y"): "y + a * dt",
  //         },
  //     )
  //     ekf = python.compile_ekf(
  //         state_model=ui_Model,
  //         process_noise=np.eye(1),
  //         sensor_models={"simple": {ui.Symbol("x"): ui.Symbol("x")}},
  //         sensor_noises={"simple": np.eye(1)},
  //         config=config,
  //     )
  //
  //     control_vector = np.array([[a]])
  //     covariance = np.eye(2)
  //     state_vector = np.array([[x, y]]).transpose()
  //
  //     if not np.isfinite(state_vector).all() or not
  //     np.isfinite(control_vector).all():
  //         # reject infinite / NaN inputs
  //         reject()
  //     if (np.abs(state_vector) > 1e100).any() or (np.abs(control_vector) >
  //     1e100).any():
  //         # reject poorly sized inputs
  //         reject()
  //
  //     next_state, next_cov = ekf.sensor_model(
  //         "simple", state=state_vector, covariance=covariance,
  //         sensor_reading=np.eye(1)
  //     )
  //     first_innovation = ekf.innovations["simple"]
  //
  //     try:
  //         starting_central_probability =
  //         multivariate_normal(cov=covariance).pdf(
  //             np.zeros_like(state_vector).transpose()
  //         )
  //         ending_central_probability = multivariate_normal(cov=next_cov).pdf(
  //             np.zeros_like(state_vector).transpose()
  //         )
  //     except np.linalg.LinAlgError:
  //         print("starting cov")
  //         print(covariance)
  //         print("next_cov")
  //         print(next_cov)
  //         raise
  //
  //     try:
  //         # more confident in state estimate after sensor update
  //         assert ending_central_probability > starting_central_probability
  //     except AssertionError:
  //         print("starting pdf")
  //         print(starting_central_probability)
  //         print("ending pdf")
  //         print(ending_central_probability)
  //         raise
  //
  //     ekf.sensor_model(
  //         "simple", state=next_state, covariance=next_cov,
  //         sensor_reading=np.eye(1)
  //     )
  //     second_innovation = ekf.innovations["simple"]
  //
  //     # state moves towards reading after a sensor update
  //     assert (
  //         np.linalg.norm(first_innovation) >
  //         np.linalg.norm(second_innovation) or
  //         np.linalg.norm(first_innovation) == 0.0
  //     )
}

}  // namespace integration
