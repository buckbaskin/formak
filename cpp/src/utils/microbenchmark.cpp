#include <formak/utils/microbenchmark.h>

#include <iomanip>  // setprecision
#include <ios>      // fixed, default_precision
#include <random>   // minstd_rand, uniform_real_distribution

namespace formak::utils {

namespace io_helpers {
std::ostream& operator<<(std::ostream& o, std::chrono::nanoseconds time) {
  auto default_precision = o.precision();
  o << std::setprecision(6);
  o << std::fixed;

  o << (time.count() / 1.0e6) << " ms";

  o << std::defaultfloat;
  o << std::setprecision(default_precision);
  return o;
}

std::ostream& operator<<(std::ostream& o,
                         const std::vector<std::chrono::nanoseconds>& times) {
  for (std::chrono::nanoseconds time : times) {
    o << time << std::endl;
  }

  return o;
}
}  // namespace io_helpers

std::vector<double> random_input(size_t size) {
  size_t seed = 1;
  std::minstd_rand prng(seed);
  std::uniform_real_distribution<> dist(0, 1);

  std::vector<double> result;

  for (size_t i = 0; i < size; i++) {
    result.push_back(dist(prng));
  }

  return result;
}

}  // namespace formak::utils
