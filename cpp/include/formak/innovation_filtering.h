#pragma once

namespace formak::innovation_filtering::edit {
template <typename InnovationT, typename CovarianceT>
bool removeInnovation(double editing_threshold, size_t reading_size,
                      const InnovationT& innovation,
                      const CovarianceT& sensor_estimate_covariance) {
  // y.T * C^{-1} * y - m > k * sqrt(2 * m)
  // y.T * C^{-1} * y > k * sqrt(2 * m) + m
  // Time Dependent > Consteval
  double normalizedInnovation =
      (innovation.transpose() * sensor_estimate_covariance.inverse() *
       innovation)(0, 0);
  double innovationExpectation =
      editing_threshold * std::sqrt(2 * reading_size) + reading_size;
  return normalizedInnovation > innovationExpectation;
}
}  // namespace formak::innovation_filtering::edit
