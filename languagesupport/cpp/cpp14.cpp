#include <gtest/gtest.h>

#include <memory>

namespace featuresupport {
namespace cpp14 {
TEST(Cpp14, MakeUnique) {
  std::unique_ptr<int> i = std::make_unique<int>(4);

  EXPECT_EQ(*i, 4);
}
}  // namespace cpp14
}  // namespace featuresupport
