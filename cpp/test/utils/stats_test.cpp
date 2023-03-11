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
  TestCovariance c;

  MultivariateNormal distribution(s, c);

  double centeredPdf = distribution.pdf(s);

  s.data[0] = 1.0;
  double offCenterPdf = distribution.pdf(s);

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

}  // namespace multivariate_normal_pdf_test
}  // namespace formak::utils::stats
