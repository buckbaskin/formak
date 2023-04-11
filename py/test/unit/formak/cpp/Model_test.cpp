#include <gtest/gtest.h>
#include <unit/simple-model.h>  // Generated

namespace unit {

TEST(Model, StateDefaultInitializationIsZero) {
  State state;
  EXPECT_EQ(state.x(), 0.0);
  EXPECT_EQ(state.y(), 0.0);
}

TEST(Model, ControlConstructor) {
  {
    Control control;
    EXPECT_DOUBLE_EQ(control.a(), 0.0);
  }

  {
    Control control({0.2});
    EXPECT_DOUBLE_EQ(control.a(), 0.2);
  }
}

TEST(Model, StateConstructor) {
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

namespace model_with_control_test {
struct Options {
  double xStart;
  double yStart;

  double xEnd;
  double yEnd;
};

class ModelWithControlTest : public ::testing::Test,
                             public ::testing::WithParamInterface<Options> {};

TEST_P(ModelWithControlTest, Test) {
  Model model;
  double dt = 0.1;

  Control control({0.2});

  State state({GetParam().xStart, GetParam().yStart});

  State next = model.model(dt, state, control);

  EXPECT_DOUBLE_EQ(next.x(), GetParam().xEnd);
  EXPECT_DOUBLE_EQ(next.y(), GetParam().yEnd);
}

INSTANTIATE_TEST_SUITE_P(StateTestCases, ModelWithControlTest,
                         ::testing::Values(Options{0.0, 0.0, 0.0, 0.02},
                                           Options{0.0, 1.0, 0.0, 1.02},
                                           Options{1.0, 0.0, 0.0, 0.02},
                                           Options{1.0, 1.0, 1.0, 1.02}));
}  // namespace model_with_control_test

namespace impl_control_test {
struct Options {
  double xStart;
  double yStart;

  double xEnd;
  double yEnd;
};

class ImplControlTest : public ::testing::Test,
                        public ::testing::WithParamInterface<Options> {};

TEST_P(ImplControlTest, Test) {
  // Generated in unit namespace
  Model model;
  double dt = 0.1;

  Control control({0.2});

  State state({GetParam().xStart, GetParam().yStart});
  auto result = model.model(dt, state, control);

  EXPECT_DOUBLE_EQ(result.x(), GetParam().xEnd);
  EXPECT_DOUBLE_EQ(result.y(), GetParam().yEnd);
}

INSTANTIATE_TEST_SUITE_P(StateTestCases, ImplControlTest,
                         ::testing::Values(Options{0.0, 0.0, 0.0, 0.02},
                                           Options{0.0, 1.0, 0.0, 1.02},
                                           Options{1.0, 0.0, 0.0, 0.02},
                                           Options{1.0, 1.0, 1.0, 1.02}));
}  // namespace impl_control_test

}  // namespace unit
