#pragma once

#include <algorithm>  // sort
#include <chrono>
#include <iterator>  // back_inserter
#include <vector>

namespace formak::utils {
template <typename InputT, typename Func>
std::vector<std::chrono::nanoseconds> microbenchmark(
    Func&& lambda, const std::vector<InputT>& inputs) {
  std::vector<std::chrono::nanoseconds> times;

  for (const InputT& i : inputs) {
    const auto start = std::chrono::steady_clock::now();
    lambda(i);
    const auto end = std::chrono::steady_clock::now();
    times.push_back(end - start);
  }
  std::sort(times.begin(), times.end());

  return times;
}
}  // namespace formak::utils
