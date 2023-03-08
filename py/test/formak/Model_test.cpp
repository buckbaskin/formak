#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>
#include <unit/simple-model.h>  // Generated

namespace integration {

RC_GTEST_PROP(CppModel, Model_impl_property, (double x, double y, double a)) {
  // def test_Model_impl_property(x, y, a):
  unit::Model model;
  double dt = 0.1;

  unit::Control control({a});
  unit::State state({x, y});

  RC_PRE(std::isfinite(x) && std::isfinite(y) && std::isfinite(a));
  RC_PRE(std::abs(x) < 1e100 && std::abs(y) < 1e100 && std::abs(a) < 1e100);

  auto next_state = model.model(dt, state, control);

  RC_ASSERT(next_state.x() == x * y);
  RC_ASSERT(next_state.y() == y + a * dt);
}

}  // namespace integration
