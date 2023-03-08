#pragma once

#include <cmath>

namespace formak::testing::stats {

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
    StateT::DataT offset = state.data - _center.data;
    Covariance::DataT inverse = _covariance.data.inverse();
    double mahalanobis_like = offset.tranpose() * inverse * offset;
    double numerator = std::exp(-1 / 2 * mahalanobis_like);

    size_t k = state.rows;
    double determinant = _covariance.determinant();
    double denominator = std::sqrt(std::pow(2 * pi, k) * determinant);

    return numerator / denominator;
  }

 private:
  StateT _center;
  CovarianceT _covariance;
};

}  // namespace formak::testing::stats
