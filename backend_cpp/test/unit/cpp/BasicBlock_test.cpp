#include <cse/ekf-cse.h>  // Generated
#include <gtest/gtest.h>
#include <no_cse/ekf-no-cse.h>  // Generated

namespace unit {

namespace cse_vs_no_cse {
struct Options {
  double xStart;
  double yStart;

  double xEnd;
  double yEnd;
};

class BasicBlockCseTest : public ::testing::Test,
                          public ::testing::WithParamInterface<Options> {};

TEST_P(BasicBlockCseTest, CSE) {
  using namespace cse;

  ExtendedKalmanFilter ekf;
  double dt = 0.1;

  Control control({0.2});

  Covariance covariance;

  State state({GetParam().xStart, GetParam().yStart});

  auto next = ekf.process_model(dt, {state, covariance}, control);

  EXPECT_DOUBLE_EQ(next.state.x(), GetParam().xEnd);
  EXPECT_DOUBLE_EQ(next.state.y(), GetParam().yEnd);
}

TEST_P(BasicBlockCseTest, NoCSE) {
  using namespace no_cse;

  ExtendedKalmanFilter ekf;
  double dt = 0.1;

  Control control({0.2});

  Covariance covariance;

  State state({GetParam().xStart, GetParam().yStart});

  auto next = ekf.process_model(dt, {state, covariance}, control);

  EXPECT_DOUBLE_EQ(next.state.x(), GetParam().xEnd);
  EXPECT_DOUBLE_EQ(next.state.y(), GetParam().yEnd);
}

INSTANTIATE_TEST_SUITE_P(StateTestCases, BasicBlockCseTest,
                         ::testing::Values(Options{0.0, 0.0, 0.0, 0.0},
                                           Options{0.0, 1.0, 0.02, 1.0},
                                           Options{1.0, 0.0, 1.0, 0.02},
                                           Options{1.0, 1.0, 1.02, 1.02}));
}  // namespace cse_vs_no_cse

}  // namespace unit
