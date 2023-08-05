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
  using formak::utils::microbenchmark;
  using namespace formak::utils::io_helpers;

  size_t size = 100;
  size_t extra_runs = 1;

  State state(StateOptions{.v = 1.0});
  formak::runtime::ManagedFilter<featuretest::ExtendedKalmanFilter> mf(
      3.0, {
               .state = state,
               .covariance = {},
           });

  Control control;

  std::vector<std::chrono::nanoseconds> manager_times = microbenchmark(
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
      tickInput(size), extra_runs);

  auto timeLog = mf.viewTimeData();
  EXPECT_EQ(timeLog.tickTimeControl.size(), size * extra_runs);
  ASSERT_EQ(timeLog.tickTimeControlReadings.size(), size * extra_runs);

  featuretest::ExtendedKalmanFilter ekf;
  double currentTime = 3.0;
  StateAndVariance combined{.state = state, .covariance = {}};

  std::vector<std::chrono::nanoseconds> no_manager_times = microbenchmark(
      [&ekf, &currentTime, &combined, &control](const TickOptions& options) {
        double process_dt = options.reading_dt_base + 0.05 - currentTime;
        combined = ekf.process_model(process_dt, combined, control);
        currentTime = currentTime + process_dt;

        featuretest::Simple zero_sensor_reading(SimpleOptions{});
        combined = ekf.sensor_model(combined, zero_sensor_reading);

        process_dt = options.reading_dt_base + 0.06 - currentTime;
        combined = ekf.process_model(process_dt, combined, control);
        currentTime = currentTime + process_dt;

        featuretest::Simple one_sensor_reading(SimpleOptions{});
        combined = ekf.sensor_model(combined, one_sensor_reading);

        process_dt = options.reading_dt_base + 0.07 - currentTime;
        combined = ekf.process_model(process_dt, combined, control);
        currentTime = currentTime + process_dt;

        featuretest::Simple two_sensor_reading(SimpleOptions{});
        combined = ekf.sensor_model(combined, two_sensor_reading);

        process_dt = options.output_dt - currentTime;
        combined = ekf.process_model(process_dt, combined, control);
        currentTime = currentTime + process_dt;
      },
      tickInput(size), extra_runs);

  double manager_p01_fastest = manager_times[1].count() / 1.0e6;
  double manager_range =
      (manager_times[size - 1] - manager_times[0]).count() / 1.0e6;
  double no_manager_p99_slowest = no_manager_times[size - 2].count() / 1.0e6;
  double no_manager_range =
      (no_manager_times[size - 1] - no_manager_times[0]).count() / 1.0e6;

  std::cout << "Manager    " << manager_p01_fastest
            << " range: " << manager_range << std::endl
            << manager_times << std::endl;
  std::cout << "No Manager " << no_manager_p99_slowest
            << " range: " << no_manager_range << std::endl
            << no_manager_times << std::endl;

  EXPECT_LT(manager_p01_fastest, no_manager_p99_slowest);
}

}  // namespace featuretest
