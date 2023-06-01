#include <example.h>
#include <gtest/gtest.h>

namespace testing {

TEST(DoesTheExampleWork, DefaultConstructor) {
  featuretests::State state;
  FAIL() << "Check this even runs";
}

}  // namespace testing
