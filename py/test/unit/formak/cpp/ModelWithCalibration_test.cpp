#include <gtest/gtest.h>
#include <unit/calibration-model.h>  // Generated

namespace unit {

namespace model_with_calibration_test {
struct Options {
  double xStart;

  double xEnd;
};

class ModelWithCalibrationTest : public ::testing::Test,
                                 public ::testing::WithParamInterface<Options> {
};

TEST_P(ModelWithCalibrationTest, Test) {
  Model model;
  double dt = 0.1;

  Calibration calibration({5.0, 0.5});

  State state({GetParam().xStart});

  State next = model.model(dt, state, calibration);

  EXPECT_DOUBLE_EQ(next.x(), GetParam().xEnd);
}

INSTANTIATE_TEST_SUITE_P(StateTestCases, ModelWithCalibrationTest,
                         ::testing::Values(Options{0.0, 5.5},
                                           Options{-1.0, 4.5}));
}  // namespace model_with_calibration_test

}  // namespace unit
