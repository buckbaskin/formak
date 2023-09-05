/// Feature Test: AST Code Generation
///
/// The code in example.h is generated via the FormaK AST tooling.
/// Passes if the test compiles without warnings and pass some basic
/// functionality in testing.

#include <example.h>
#include <gtest/gtest.h>

namespace testing {

TEST(DoesTheExampleWork, DefaultConstructor) {
  featuretest::State state;

  EXPECT_DOUBLE_EQ(state.CON_pos_pos_x(), 0.0);
  EXPECT_DOUBLE_EQ(state.CON_pos_pos_y(), 0.0);
}

TEST(DoesTheExampleWork, Constructor) {
  featuretest::State state({.CON_pos_pos_x = 1.0, .CON_pos_pos_y = -2.0});

  state.CON_pos_pos_x() = 10.0;

  EXPECT_DOUBLE_EQ(state.CON_pos_pos_x(), 10.0);
  EXPECT_DOUBLE_EQ(state.CON_pos_pos_y(), -2.0);
}

TEST(DoesTheExampleWork, Function) {
  const featuretest::State lower({
      .CON_pos_pos_x = 1.0,
      .CON_pos_pos_y = -2.0,
      .CON_pos_pos_z = 1.0,
  });
  const featuretest::State upper({
      .CON_pos_pos_x = 5.0,
      .CON_pos_pos_y = -1.0,
      .CON_pos_pos_z = 2.0,
  });

  const featuretest::State value({
      .CON_pos_pos_x = 0.0,
      .CON_pos_pos_y = -4.0,
      .CON_pos_pos_z = 3.0,
  });

  featuretest::State clamped = elementwise_clamp(value, lower, upper);

  EXPECT_DOUBLE_EQ(clamped.CON_pos_pos_x(), 1.0);
  EXPECT_DOUBLE_EQ(clamped.CON_pos_pos_y(), -2.0);
  EXPECT_DOUBLE_EQ(clamped.CON_pos_pos_z(), 2.0);
}

}  // namespace testing
