#include <cse/cse-model.h>  // Generated
#include <formak/utils/microbenchmark.h>
#include <gtest/gtest.h>
#include <no_cse/no-cse-model.h>  // Generated

#include <chrono>
#include <iterator>  // back_inserter
#include <ostream>   // cout
#include <random>
#include <tuple>
#include <vector>

namespace {
std::ostream& operator<<(std::ostream& o,
                         std::vector<std::chrono::nanoseconds> times) {
  for (std::chrono::nanoseconds time : times) {
    o << (time.count() / 1.0e6) << " ms" << std::endl;
  }
  return o;
}
}  // namespace

namespace featuretest {

TEST(CppModel, Simple) {
  using formak::utils::microbenchmark;

  cse::Model cse_model;
  no_cse::Model no_cse_model;

  auto input_base = ([]() {
    size_t seed = 1;
    std::minstd_rand prng(seed);
    std::uniform_real_distribution<> dist(0, 1);

    std::vector<std::tuple<double, double>> result;

    for (int i = 0; i < 100; i++) {
      result.push_back(std::make_tuple(dist(prng), dist(prng)));
    }

    return result;
  })();

  std::vector<cse::StateOptions> cse_input;
  std::transform(input_base.cbegin(), input_base.cend(),
                 std::back_inserter(cse_input),
                 [](std::tuple<double, double> t) {
                   const auto& [left, right] = t;
                   return cse::StateOptions{
                       .left = left,
                       .right = right,
                   };
                 });
  std::vector<no_cse::StateOptions> no_cse_input;
  std::transform(input_base.cbegin(), input_base.cend(),
                 std::back_inserter(no_cse_input),
                 [](std::tuple<double, double> t) {
                   const auto& [left, right] = t;
                   return no_cse::StateOptions{
                       .left = left,
                       .right = right,
                   };
                 });

  std::vector<std::chrono::nanoseconds> cse_times = microbenchmark(
      [&cse_model](const cse::StateOptions& options) {
        cse_model.model(0.1, options);
      },
      cse_input);
  std::vector<std::chrono::nanoseconds> no_cse_times = microbenchmark(
      [&no_cse_model](const no_cse::StateOptions& options) {
        no_cse_model.model(0.1, options);
      },
      no_cse_input);

  double cse_p99_slowest = cse_times[98].count() / 1.0e6;
  double no_cse_p01_fastest = no_cse_times[1].count() / 1.0e6;
  std::cout << "CSE    " << cse_p99_slowest << std::endl
            << cse_times << std::endl;
  std::cout << "No CSE " << no_cse_p01_fastest << std::endl
            << no_cse_times << std::endl;

  EXPECT_LT(cse_p99_slowest, no_cse_p01_fastest);
}

}  // namespace featuretest
