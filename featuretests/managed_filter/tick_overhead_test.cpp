#include <featuretest/example.h>  // Generated
#include <formak/runtime/ManagedFilter.h>
#include <formak/utils/microbenchmark.h>
#include <gtest/gtest.h>

#include <vector>

namespace testing {
using formak::runtime::ManagedFilter;

TEST(ManagedFilterTickTest, MultipleReadings) {
  using namespace featuretest;
  State state(StateOptions{.v = 1.0});
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      3.0, {
               .state = state,
               .covariance = {},
           });

  Control control;

  std::vector<std::chrono::nanoseconds> manager_times = microbenchmark(
      [&mf]() {
        auto state0p1 = ([&mf, &control]() {
          return mf.tick(3.1, control,
                         {
                             mf.wrap<Simple>(3.05, SimpleOptions{}),
                             mf.wrap<Simple>(3.06, SimpleOptions{}),
                             mf.wrap<Simple>(3.07, SimpleOptions{}),
                         });
        })();

        auto state0p2 = ([&mf, &control]() {
          return mf.tick(3.2, control,
                         {
                             mf.wrap<Simple>(3.15, SimpleOptions{}),
                             mf.wrap<Accel>(3.16, AccelOptions{.a = 1.0}),
                             mf.wrap<Simple>(3.17, SimpleOptions{}),
                         });
        })();
      },
      manager_input);

  std::vector<std::chrono::nanoseconds> no_manager_times = microbenchmark(
      [&no_manager_model](const no_manager::StateOptions& options) {
        no_manager_model.model(0.1, options);
      },
      no_manager_input);

  double manager_p01_fastest = manager_times[1].count() / 1.0e6;
  double no_manager_p99_slowest = no_manager_times[98].count() / 1.0e6;

  std::cout << "Manager    " << manager_p01_fastest << std::endl
            << manager_times << std::endl;
  std::cout << "No Manager " << no_manager_p99_slowest << std::endl
            << no_manager_times << std::endl;

  EXPECT_LT(manager_p01_fastest, no_manager_p99_slowest);
}

}  // namespace testing
