#include <featuretest/example.h>  // Generated
#include <formak/runtime/ManagedFilter.h>
#include <formak/utils/microbenchmark.h>
#include <gtest/gtest.h>

#include <vector>

namespace featuretest {
using formak::runtime::ManagedFilter;

struct TickOptions {
  double output_dt;
  double reading_dt_base;
};

std::vector<TickOptions> tickInput(size_t size) {
  constexpr double dt = 0.1;
  std::vector<TickOptions> result;
  for (size_t i = 0; i < size; ++i) {
    result.push_back(TickOptions{3.1 + i * dt, 3.0 + i * dt});
  }
  return result;
}

TEST(ManagedFilterTickTest, MultipleReadings) {
  // using formak::utils::io_helpers;

  State state(StateOptions{.v = 1.0});
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      3.0, {
               .state = state,
               .covariance = {},
           });

  Control control;

  std::vector<std::chrono::nanoseconds> manager_times =
      formak::utils::microbenchmark(
          [&mf, &control](const TickOptions& options) {
            mf.tick(options.output_dt, control,
                    {
                        mf.wrap<Simple>(options.reading_dt_base + 0.05,
                                        SimpleOptions{}),
                        mf.wrap<Simple>(options.reading_dt_base + 0.06,
                                        SimpleOptions{}),
                        mf.wrap<Simple>(options.reading_dt_base + 0.07,
                                        SimpleOptions{}),
                    });
          },
          tickInput(100));

  std::vector<std::chrono::nanoseconds> no_manager_times = microbenchmark(
      [&no_manager_model](const TickOptions& options) {
        no_manager_model.model(0.1, options);
      },
      tickInput(100));

  double manager_p01_fastest = manager_times[1].count() / 1.0e6;
  double no_manager_p99_slowest = no_manager_times[98].count() / 1.0e6;

  // std::cout << "Manager    " << manager_p01_fastest << std::endl
  //           << manager_times << std::endl;
  // std::cout << "No Manager " << no_manager_p99_slowest << std::endl
  //           << no_manager_times << std::endl;

  EXPECT_LT(manager_p01_fastest, no_manager_p99_slowest);
}

}  // namespace featuretest
