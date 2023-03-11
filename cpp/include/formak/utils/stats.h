#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace formak::utils::stats {

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
bool IsPositiveDefinite(const Eigen::Matrix<Scalar, RowsAtCompileTime,
                                            ColsAtCompileTime>& covariance) {
  Eigen::LLT<
      typename Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
      llt(covariance);
  return llt.info() == Eigen::Success;
}

template <typename StateT, typename CovarianceT>
class MultivariateNormal {
 public:
  MultivariateNormal(const StateT& center, const CovarianceT& covariance)
      : _center(center), _covariance(covariance) {
  }

  /// Example usage:
  ///
  /// double starting_central_probability = MultivariateNormal(covariance).pdf(
  ///     state_vector
  /// )
  ///
  /// Note: Assumes covariance is positive definite
  double pdf(const StateT& state) {
    typename StateT::DataT offset = state.data - _center.data;
    typename CovarianceT::DataT inverse = _covariance.data.inverse();
    double mahalanobis_like = offset.transpose() * inverse * offset;
    double numerator = std::exp(-1 / 2 * mahalanobis_like);

    size_t k = state.rows;
    double determinant = _covariance.data.determinant();
    double denominator = std::sqrt(std::pow(2 * M_PI, k) * determinant);

    return numerator / denominator;
  }

 private:
  StateT _center;
  CovarianceT _covariance;
};

}  // namespace formak::utils::stats
