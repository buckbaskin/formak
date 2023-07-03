#include <gtest/gtest.h>

#include <memory>

namespace featuresupport {
namespace cpp17 {
TEST(Cpp17, IfConstexpr) {
  if constexpr (true) {
    SUCCEED();
  } else {
    FAIL();
  }
}
}  // namespace cpp17
}  // namespace featuresupport
