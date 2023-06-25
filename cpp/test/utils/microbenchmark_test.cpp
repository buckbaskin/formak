#include <formak/utils/microbenchmark.h>
#include <gtest/gtest.h>

#include <cmath>

namespace formak::utils {
namespace interface_test {
using formak::utils::microbenchmark;

TEST(MicrobenchmarkTest, Zero) {
  auto results =
      microbenchmark([](double d) { return sqrt(d); }, std::vector<double>{});
  EXPECT_EQ(results.size(), 0);
}

TEST(MicrobenchmarkTest, One) {
  std::vector<double> in;
  in.push_back(1.0);
  auto results = microbenchmark([](double d) { return sqrt(d); }, in);

  ASSERT_EQ(results.size(), 1);

  for (const auto& time : results) {
    EXPECT_GT(time, std::chrono::nanoseconds(0));
  }
}

TEST(MicrobenchmarkTest, Many) {
  std::vector<double> in;
  in.push_back(1.0);
  in.push_back(-1.0);
  in.push_back(std::nan("test"));

  auto results = microbenchmark([](double d) { return sqrt(d); }, in);

  ASSERT_EQ(results.size(), 3);

  for (const auto& time : results) {
    EXPECT_GT(time, std::chrono::nanoseconds(0));
  }
}

}  // namespace interface_test

}  // namespace formak::utils
