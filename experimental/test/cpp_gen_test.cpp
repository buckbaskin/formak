#include <gtest/gtest.h>
#include <jinja_basic_class.h>

namespace experimental {

TEST(JinjaGeneratedTest, TwoMethods) {
  Stateful s;

  EXPECT_EQ(s.getValue(), 0);

  s.update();

  EXPECT_EQ(s.getValue(), 1);
}
}  // namespace experimental
