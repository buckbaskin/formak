#include <gtest/gtest.h>
#include <unit/simple-model.h>  // Generated

namespace unit {

TEST(CppModel, impl_control) {
  // Generated in unit namespace
  Model model;
  double dt = 0.1;

  Control control({0.2});

  {
    State state({0.0, 0.0});
    auto result = model.process_model(dt, state, control_vector);

    EXPECT_DOUBLE_EQ(result.x, 0.0);
    EXPECT_DOUBLE_EQ(result.y, 0.02);
  }

  {
    State state({0.0, 1.0});
    auto result = model.process_model(dt, state, control_vector);

    EXPECT_DOUBLE_EQ(result.x, 0.0);
    EXPECT_DOUBLE_EQ(result.y, 1.02);
  }

  {
    State state({1.0, 0.0});
    auto result = model.process_model(dt, state, control_vector);

    EXPECT_DOUBLE_EQ(result.x, 0.0);
    EXPECT_DOUBLE_EQ(result.y, 0.02);
  }

  {
    State state({1.0, 1.0});
    auto result = model.process_model(dt, state, control_vector);

    EXPECT_DOUBLE_EQ(result.x, 1.0);
    EXPECT_DOUBLE_EQ(result.y, 1.02);
  }
}

}  // namespace unit
