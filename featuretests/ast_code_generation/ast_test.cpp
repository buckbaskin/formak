#include <example.h>
#include <gtest/gtest.h>

namespace testing {

TEST(DoesTheExampleWork, DefaultConstructor) {
  featuretest::State state;
  FAIL() << "Check this even runs";
}

}  // namespace testing
