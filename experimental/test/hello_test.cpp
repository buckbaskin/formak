#include <experimental/hello_greet.h>
#include <gtest/gtest.h>

namespace {

TEST(FirstTest, Magic) {
  EXPECT_EQ(magic(), 1);
}

}  // namespace
