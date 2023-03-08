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
}

}  // namespace integration
