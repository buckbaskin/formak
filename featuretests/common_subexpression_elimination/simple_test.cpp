#include <cse/cse-model.h>  // Generated
#include <gtest/gtest.h>
#include <no_cse/no-cse-model.h>  // Generated

#include <algorithm>
#include <chrono>
#include <random>
#include <tuple>

namespace featuretest {

TEST(CppModel, Simple) {
  cse::Model cse_model;
  no_cse::Model no_cse_model;

  no_cse_model.model(0.1, no_cse::StateOptions{.left = 1.0, .right = 0.1});

  auto input = ([]() {
    size_t seed = 1;
    std::minstd_rand prng(seed);
    std::uniform_real_distribution<> dist(0, 1);

    std::vector<std::tuple<double, double>> result;

    for (int i = 0; i < 100; i++) {
      result.push_back(std::make_tuple(dist(prng), dist(prng)));
    }

    return result;
  })();

  std::vector<std::chrono::nanoseconds> cse_times;
  std::vector<std::chrono::nanoseconds> no_cse_times;

  for (const auto& [left, right] : input) {
    const auto cse_start = std::chrono::system_clock::now();
    cse_model.model(0.1, cse::StateOptions{.left = left, .right = right});
    const auto cse_end = std::chrono::system_clock::now();
    cse_times.push_back(cse_end - cse_start);
  }
  std::sort(cse_times.begin(), cse_times.end());
  double cse_p99_slowest = cse_times[98].count() / 1.0e6;

  for (const auto& [left, right] : input) {
    const auto no_cse_start = std::chrono::system_clock::now();
    no_cse_model.model(0.1, no_cse::StateOptions{.left = left, .right = right});
    const auto no_cse_end = std::chrono::system_clock::now();
    no_cse_times.push_back(no_cse_end - no_cse_start);
  }
  std::sort(no_cse_times.begin(), no_cse_times.end());
  double no_cse_p01_fastest = no_cse_times[1].count() / 1.0e6;

  for (const auto& cse_time : cse_times) {
    std::cout << "cse time " << cse_time.count() / 1.0e6 << " ms" << std::endl;
  }

  for (const auto& no_cse_time : no_cse_times) {
    std::cout << "no_cse time " << no_cse_time.count() / 1.0e6 << " ms"
              << std::endl;
  }

  EXPECT_LT(cse_p99_slowest, no_cse_p01_fastest);
}

}  // namespace featuretest
