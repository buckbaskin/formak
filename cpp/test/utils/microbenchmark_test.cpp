#include <formak/utils/microbenchmark.h>
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>  // setprecision
#include <ios>      // fixed, default_precision

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

  EXPECT_GT(results[results.size() - 1], results[0]);
}
}  // namespace interface_test

namespace trig_timing_test {

TEST(TrigTimingTest, OneHundredCompare) {
  std::vector<double> in = formak::utils::random_input(10100);

  auto sum_version = microbenchmark(
      [](double x) {
        double y = 1.0;
        return sin(x + y);
      },
      in);
  auto expand_version = microbenchmark(
      [](double x) {
        double y = 1.0;
        return sin(x) * cos(y) + sin(y) * cos(x);
      },
      in);

  size_t p01 = static_cast<size_t>(round(in.size() / 100.0 * 1.0));
  size_t p99 = static_cast<size_t>(round(in.size() / 100.0 * 99.0));

  std::cout << std::fixed << std::setprecision(6) << "Results: Sum    "
            << sum_version[p99].count() / 1e6 << " ms Expand "
            << expand_version[p01].count() / 1e6 << std::endl;
  // Better?
  EXPECT_LT(sum_version[p99], expand_version[p01]);
}

}  // namespace trig_timing_test

}  // namespace formak::utils
