#include <formak/runtime/ManagedFilter.h>
#include <gtest/gtest.h>

namespace unit {

struct StateAndVariance {
  double state = 0.0;
  double covariance = 0.0;
};
// TODO(buck): Do all filters have a control?
struct Control {};

struct TestImpl {
  using StateAndVarianceT = StateAndVariance;
  using ControlT = Control;
};

TEST(ManagedFilter, Constructor) {
  // template <typename Impl>
  // class ManagedFilter {
  //  public:
  //   ManagedFilter(double initialTimestamp,
  //                 const typename Impl::StateAndVarianceT& initialState)
  //       : _currentTime(initialTimestamp), _state(initialState) {
  //   }
  // }

  // [[maybe_unused]] because this test is focused on the constructor only.
  // Passes if construction and deconstruction are successful
  [[maybe_unused]] formak::runtime::ManagedFilter<TestImpl> mf(
      1.23, StateAndVariance{.state = 4.0, .covariance = 1.0});

  // Note(buck): This test is helpful (compiler error because _impl isn't
  // initialized...)
}

}  // namespace unit
