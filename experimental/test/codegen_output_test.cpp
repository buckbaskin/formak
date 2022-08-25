#include <codegen/codegen.h>
#include <gtest/gtest.h>

namespace {
TEST(Codegen, Selection) {
  EXPECT_EQ(selected(), 2);
}
}  // namespace
