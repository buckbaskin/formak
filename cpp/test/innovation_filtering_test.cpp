#include <formak/innovation_filtering.h>
#include <gtest/gtest.h>

namespace formak::innovation_filtering {

TEST(EditDistance, SingleInnovation) {
  constexpr size_t size = 1;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 1.0;
  InnovationT innovation = InnovationT::Zero();
  CovarianceT covariance = CovarianceT::Identity();

  bool result =
      edit::removeInnovation(editing_threshold, innovation, covariance);
  EXPECT_FALSE(result);

  innovation(0, 0) = 2.0;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance));
}

TEST(EditDistance, DoubleInnovation) {
  constexpr size_t size = 2;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 1.0;
  InnovationT innovation = InnovationT::Zero();
  CovarianceT covariance = CovarianceT::Identity();

  EXPECT_FALSE(
      edit::removeInnovation(editing_threshold, innovation, covariance));

  innovation(0, 0) = 2.0;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance));
}

TEST(EditDistance, TripleInnovation) {
  constexpr size_t size = 3;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 1.0;
  InnovationT innovation = InnovationT::Zero();
  CovarianceT covariance = CovarianceT::Identity();

  EXPECT_FALSE(
      edit::removeInnovation(editing_threshold, innovation, covariance));

  innovation(0, 0) = 2.0;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance));
}

}  // namespace formak::innovation_filtering
