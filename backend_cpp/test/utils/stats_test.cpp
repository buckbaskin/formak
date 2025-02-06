#include <formak/utils/stats.h>
#include <gtest/gtest.h>

namespace formak::utils::stats {
namespace is_positive_definite_test {
using ExampleT = Eigen::Matrix<double, 3, 3>;

TEST(IsPositiveDefiniteTest, Positive) {
  ExampleT example = ExampleT::Identity();

  EXPECT_TRUE(IsPositiveDefinite(example));

  example(2, 1) = 0.5;
  example(1, 2) = example(2, 1);

  EXPECT_TRUE(IsPositiveDefinite(example));

  example(2, 1) = -0.5;
  example(1, 2) = example(2, 1);

  EXPECT_TRUE(IsPositiveDefinite(example));
}

TEST(IsPositiveDefiniteTest, Negative) {
  ExampleT asymmetric = ExampleT::Identity();
  asymmetric(2, 0) = 1.0;

  EXPECT_FALSE(IsPositiveDefinite(asymmetric));

  ExampleT zero = ExampleT::Zero();

  EXPECT_FALSE(IsPositiveDefinite(zero));

  ExampleT negative = ExampleT::Identity();
  negative(1, 1) = -1.0;

  EXPECT_FALSE(IsPositiveDefinite(negative));
}
}  // namespace is_positive_definite_test

namespace multivariate_normal_pdf_test {
struct TestState {
  static constexpr size_t rows = 3;
  using DataT = Eigen::Matrix<double, rows, 1>;

  DataT data = DataT::Zero();
};
struct TestCovariance {
  using DataT = Eigen::Matrix<double, 3, 3>;

  DataT data = DataT::Identity();
};

TEST(MultivariteNormalPdfTest, BasicInDistribution) {
  TestState s;
  TestState _center = s;
  TestCovariance c;

  MultivariateNormal distribution(s, c);

  double centeredPdf = distribution.pdf(s);

  {
    typename TestState::DataT offset = s.data - _center.data;
    std::cout << "offset 1 " << std::endl << offset << std::endl;
    typename TestCovariance::DataT inverse = c.data.inverse();
    double mahalanobis_like = offset.transpose() * inverse * offset;

    std::cout << "mahalanobis_like " << mahalanobis_like << std::endl;
    double numerator = std::exp(-0.5 * mahalanobis_like);

    std::cout << "numerator " << numerator << std::endl;

    size_t k = TestState::rows;
    double determinant = c.data.determinant();
    double denominator = std::sqrt(std::pow(2 * M_PI, k) * determinant);

    std::cout << "result " << (numerator / denominator) << std::endl;
  }

  s.data(0, 0) = 1.0;
  double offCenterPdf = distribution.pdf(s);

  {
    typename TestState::DataT offset = s.data - _center.data;
    std::cout << "offset 2 " << std::endl
              << (s.data - _center.data) << std::endl;
    typename TestCovariance::DataT inverse = c.data.inverse();
    double mahalanobis_like = offset.transpose() * inverse * offset;

    std::cout << "mahalanobis_like " << mahalanobis_like << std::endl;

    double numerator = std::exp(-0.5 * mahalanobis_like);

    std::cout << "numerator " << numerator << std::endl;

    size_t k = TestState::rows;
    double determinant = c.data.determinant();
    double denominator = std::sqrt(std::pow(2 * M_PI, k) * determinant);

    std::cout << "result " << (numerator / denominator) << std::endl;
  }

  EXPECT_GT(centeredPdf, offCenterPdf);
}

TEST(MultivariteNormalPdfTest, BasicInDistributionWithMoreVariance) {
  TestState s;
  TestCovariance c;
  c.data *= 2;

  MultivariateNormal distribution(s, c);

  double centeredPdf = distribution.pdf(s);

  s.data[0] = 1.0;
  double offCenterPdf = distribution.pdf(s);

  EXPECT_GT(centeredPdf, offCenterPdf);
}

TEST(MultivariteNormalPdfTest, BasicInDistributionWithLessVariance) {
  TestState s;
  TestCovariance c;
  c.data *= 0.5;

  MultivariateNormal distribution(s, c);

  double centeredPdf = distribution.pdf(s);

  s.data[0] = 1.0;
  double offCenterPdf = distribution.pdf(s);

  EXPECT_GT(centeredPdf, offCenterPdf);
}

TEST(MultivariteNormalPdfTest, AcrossVariance) {
  TestState s;

  double lessVariance = ([&s]() {
    TestCovariance c;
    c.data *= 0.5;
    return MultivariateNormal(s, c).pdf(s);
  })();
  double middleVariance = ([&s]() {
    TestCovariance c;
    return MultivariateNormal(s, c).pdf(s);
  })();
  double moreVariance = ([&s]() {
    TestCovariance c;
    c.data *= 2.0;
    return MultivariateNormal(s, c).pdf(s);
  })();

  EXPECT_GT(lessVariance, middleVariance);
  EXPECT_GT(middleVariance, moreVariance);
}

TEST(MultivariteNormalPdfTest, GoldenValues) {
  // >>> from scipy.stats import multivariate_normal
  // >>> import numpy as np
  // >>> cov = np.eye(2)
  // >>> cov[0,0] = 2.5e-7
  // >>> cov[1,0] = 2.5e-5
  // >>> cov[0,1] = 2.5e-5
  // >>> cov[1,1] = 1.0025
  // >>> multivariate_normal(cov=cov).pdf(np.array([[0,0]]))
  // 318.309886183791
  struct TestState {
    const size_t rows = 2;
    using DataT = Eigen::Matrix<double, 2, 1>;

    DataT data = DataT::Zero();
  };
  struct TestCovariance {
    const size_t rows = 2;
    using DataT = Eigen::Matrix<double, 2, 2>;

    DataT data = DataT::Identity();
  };

  TestState zero;
  TestCovariance cov;
  cov.data(0, 0) = 2.5e-7;
  cov.data(1, 0) = 2.5e-5;
  cov.data(0, 1) = 2.5e-5;
  cov.data(1, 1) = 1.0025;

  EXPECT_NEAR(MultivariateNormal(zero, cov).pdf(zero), 318.309886183791, 1e-12);

  // >>> from scipy.stats import multivariate_normal
  // >>> import numpy as np
  // >>> cov = np.eye(2)
  // >>> multivariate_normal(cov=cov).pdf(np.array([[0,0]]))
  // 0.15915494309189535

  cov.data = TestCovariance::DataT::Identity();
  EXPECT_NEAR(MultivariateNormal(zero, cov).pdf(zero), 0.15915494309189535,
              1e-12);
}

}  // namespace multivariate_normal_pdf_test
}  // namespace formak::utils::stats
