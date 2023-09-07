#include <formak/innovation_filtering.h>
#include <gtest/gtest.h>

namespace formak::innovation_filtering {

TEST(EditDistance, SingleInnovation) {
  constexpr size_t size = 1;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 1.0;
  InnovationT innovation = InnovationT::Zero();
  CovarianceT covariance_inv = CovarianceT::Identity();

  bool result =
      edit::removeInnovation(editing_threshold, innovation, covariance_inv);
  EXPECT_FALSE(result);

  innovation(0, 0) = 2.0;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));
}

TEST(EditDistance, DoubleInnovation) {
  constexpr size_t size = 2;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 1.0;
  // y.T * C^{-1} * y - m > k * sqrt(2 * m)
  // y * 1 * y - 2 > 1.0 * sqrt(2 * 2)
  InnovationT innovation = InnovationT::Zero();
  innovation(0, 0) = sqrt(sqrt(2 * size) + size) - 0.01;

  CovarianceT covariance_inv = CovarianceT::Identity();

  EXPECT_FALSE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));

  innovation(0, 0) = sqrt(sqrt(2 * size) + size) + 0.01;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));
}

TEST(EditDistance, TripleInnovation) {
  constexpr size_t size = 3;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 1.0;
  InnovationT innovation = InnovationT::Zero();
  innovation(0, 0) = sqrt(sqrt(2 * size) + size) - 0.01;

  CovarianceT covariance_inv = CovarianceT::Identity();

  EXPECT_FALSE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));

  innovation(0, 0) = sqrt(sqrt(2 * size) + size) + 0.01;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));
}

TEST(EditDistance, EditThreshold1) {
  constexpr size_t size = 1;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 1.0;
  // y.T * C^{-1} * y - m > k * sqrt(2 * m)
  // y * 1 * y - 2 > 1.0 * sqrt(2 * 2)
  InnovationT innovation = InnovationT::Zero();
  innovation(0, 0) = sqrt(sqrt(2 * size) + size) - 0.01;

  CovarianceT covariance_inv = CovarianceT::Identity();

  EXPECT_FALSE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));

  innovation(0, 0) = sqrt(sqrt(2 * size) + size) + 0.01;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));
}

TEST(EditDistance, EditThreshold3) {
  constexpr size_t size = 1;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 3.0;
  InnovationT innovation = InnovationT::Zero();
  innovation(0, 0) = sqrt(editing_threshold * sqrt(2 * size) + size) - 0.01;

  CovarianceT covariance_inv = CovarianceT::Identity();

  EXPECT_FALSE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));

  innovation(0, 0) = sqrt(editing_threshold * sqrt(2 * size) + size) + 0.01;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));
}

TEST(EditDistance, EditThreshold5) {
  constexpr size_t size = 1;
  using InnovationT = Eigen::Matrix<double, size, 1>;
  using CovarianceT = Eigen::Matrix<double, size, size>;

  double editing_threshold = 5.0;
  InnovationT innovation = InnovationT::Zero();
  innovation(0, 0) = sqrt(editing_threshold * sqrt(2 * size) + size) - 0.01;

  CovarianceT covariance_inv = CovarianceT::Identity();

  EXPECT_FALSE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));

  innovation(0, 0) = sqrt(editing_threshold * sqrt(2 * size) + size) + 0.01;

  EXPECT_TRUE(
      edit::removeInnovation(editing_threshold, innovation, covariance_inv));
}

}  // namespace formak::innovation_filtering
