#pragma once

#include <algorithm>  // sort
#include <chrono>
#include <ostream>
#include <vector>

namespace formak::utils {

namespace io_helpers {
std::ostream& operator<<(std::ostream& o, std::chrono::nanoseconds time);

std::ostream& operator<<(std::ostream& o,
                         const std::vector<std::chrono::nanoseconds>& times);
}  // namespace io_helpers

std::vector<double> random_input(size_t size);

/// \brief Time the lambda function for each input
template <typename InputT, typename Func>
std::vector<std::chrono::nanoseconds> microbenchmark(
    Func&& lambda, const std::vector<InputT>& inputs, size_t count = 128) {
  std::vector<std::chrono::nanoseconds> times;

  for (const InputT& i : inputs) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t j = 0; j < count; ++j) {
      lambda(i);
    }
    const auto end = std::chrono::steady_clock::now();
    times.push_back(end - start);
  }
  std::sort(times.begin(), times.end());

  return times;
}

}  // namespace formak::utils
