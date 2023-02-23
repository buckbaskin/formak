#include <gtest/gtest.h>
#include <unit/simple-model.h>  // Generated

namespace unit {

TEST(Model, State_default_initialization_is_zero) {
  State state;
  EXPECT_EQ(state.x(), 0.0);
  EXPECT_EQ(state.y(), 0.0);
}

TEST(Model, Control_constructor) {
  {
    Control control;
    EXPECT_DOUBLE_EQ(control.a(), 0.0);
  }

  {
    Control control({0.2});
    EXPECT_DOUBLE_EQ(control.a(), 0.2);
  }
}

TEST(Model, State_constructor) {
  {
    State state;
    EXPECT_DOUBLE_EQ(state.x(), 0.0);
    EXPECT_DOUBLE_EQ(state.y(), 0.0);
  }

  {
    State state({0.2, -1.6});
    EXPECT_DOUBLE_EQ(state.x(), 0.2);
    EXPECT_DOUBLE_EQ(state.y(), -1.6);
  }
}

TEST(Model, model_with_control) {
  Model model;
  double dt = 0.1;

  Control control({0.2});

  {
    State state({0.0, 0.0});

    State next = model.model(dt, state, control);

    EXPECT_DOUBLE_EQ(next.x(), 0.0);
    EXPECT_DOUBLE_EQ(next.y(), 0.02);
  }

  {
    State state({0.0, 1.0});

    State next = model.model(dt, state, control);

    EXPECT_DOUBLE_EQ(next.x(), 0.0);
    EXPECT_DOUBLE_EQ(next.y(), 1.02);
  }

  {
    State state({1.0, 0.0});
    EXPECT_DOUBLE_EQ(state.x(), 1.0);
    EXPECT_DOUBLE_EQ(state.y(), 0.0);

    State next = model.model(dt, state, control);

    EXPECT_DOUBLE_EQ(next.x(), 0.0);
    EXPECT_DOUBLE_EQ(next.y(), 0.02);
  }

  {
    State state({1.0, 1.0});

    State next = model.model(dt, state, control);

    EXPECT_DOUBLE_EQ(next.x(), 1.0);
    EXPECT_DOUBLE_EQ(next.y(), 1.02);
  }
}

TEST(CppModel, impl_control) {
  // Generated in unit namespace
  Model model;
  double dt = 0.1;

  Control control({0.2});

  {
    State state({0.0, 0.0});
    auto result = model.model(dt, state, control);

    EXPECT_DOUBLE_EQ(result.x(), 0.0);
    EXPECT_DOUBLE_EQ(result.y(), 0.02);
  }

  {
    State state({0.0, 1.0});
    auto result = model.model(dt, state, control);

    EXPECT_DOUBLE_EQ(result.x(), 0.0);
    EXPECT_DOUBLE_EQ(result.y(), 1.02);
  }

  {
    State state({1.0, 0.0});
    auto result = model.model(dt, state, control);

    EXPECT_DOUBLE_EQ(result.x(), 0.0);
    EXPECT_DOUBLE_EQ(result.y(), 0.02);
  }

  {
    State state({1.0, 1.0});
    auto result = model.model(dt, state, control);

    EXPECT_DOUBLE_EQ(result.x(), 1.0);
    EXPECT_DOUBLE_EQ(result.y(), 1.02);
  }
}

// TODO(buck): Property testing
// @given(floats(), floats(), floats())
// @settings(deadline=timedelta(seconds=2))
// def test_Model_impl_property(x, y, a):

}  // namespace unit
