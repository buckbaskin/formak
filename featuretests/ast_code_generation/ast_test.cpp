#include <example.h>
#include <gtest/gtest.h>

namespace testing {

TEST(DoesTheExampleWork, DefaultConstructor) {
  featuretest::State state;
}

TEST(DoesTheExampleWork, Constructor) {
  featuretest::State state({.CON_pos_pos_x = 1.0, .CON_pos_pos_y = -2.0});

  state.CON_pos_pos_x() = 10.0;

  EXPECT_DOUBLE_EQ(state.CON_pos_pos_x(), 10.0);
  EXPECT_DOUBLE_EQ(state.CON_pos_pos_y(), -2.0);
}

TEST(DoesTheExampleWork, Constructor) {
  const featuretest::State lower(
      {.CON_pos_pos_x = 1.0, .CON_pos_pos_y = -2.0, .CON_pos_pos_z = 2.0});
  const featuretest::State upper(
      {.CON_pos_pos_x = 5.0, .CON_pos_pos_y = -1.0, .CON_pos_pos_z = 1.0});

  featuretest::State value(
      {.CON_pos_pos_x = 0.0, .CON_pos_pos_y = -4.0, .CON_pos_pos_z = 3.0});

  featuretest::State clamped = elementwise_clamp(lower, value, upper);

  EXPECT_DOUBLE_EQ(clamped.CON_pos_pos_x, 0.0);
  EXPECT_DOUBLE_EQ(clamped.CON_pos_pos_y, -2.0);
  EXPECT_DOUBLE_EQ(clamped.CON_pos_pos_z, 2.0);
}

}  // namespace testing
