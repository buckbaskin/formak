#include <gtest/gtest.h>

namespace languagesupport {
namespace cpp11 {

TEST(Cpp11, AutoAndIsSame) {
  int i = 0;
  auto a = i;
  static_assert(std::is_same_v<decltype(a), int> == true);
  EXPECT_EQ(a, 0);
}

}  // namespace cpp11
}  // namespace languagesupport
