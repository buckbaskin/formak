#pragma once
#include <Eigen/Dense>
#include <cmath>    // sqrt
#include <cstddef>  // size_t

namespace formak::innovation_filtering::edit {
template <int reading_size>
bool removeInnovation(double editing_threshold,
                      const Eigen::Matrix<double, reading_size, 1>& innovation,
                      const Eigen::Matrix<double, reading_size, reading_size>&
                          sensor_estimate_covariance) {
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
