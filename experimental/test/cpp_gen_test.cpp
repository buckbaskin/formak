#include <gtest/gtest.h>
#include <jinja_basic_class.h>

namespace experimental {

struct TestModelInput {
  double x;
  double y;
};

class JinjaGeneratedTest : public ::testing::TestWithParam<TestModelInput> {};

TEST(JinjaGeneratedTest, TwoMethods) {
  Stateful s;

  EXPECT_EQ(s.getValue(), 0);

  s.update();

  EXPECT_EQ(s.getValue(), 1);
}

TEST_P(JinjaGeneratedTest, SympyModel) {
  // x*y + x + y + 1
  SympyModel m;

  {
    double x = GetParam().x;
    double y = GetParam().y;
    EXPECT_DOUBLE_EQ(m.model(x, y), x * y + x + y + 1);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MatchModelAcrossLanguages, JinjaGeneratedTest,
    testing::Values(TestModelInput{0, 0}, TestModelInput{1, 0},
                    TestModelInput{0, 1}, TestModelInput{-1, 1},
                    TestModelInput{5, 7}, TestModelInput{-10, 4e-6}));

}  // namespace experimental
