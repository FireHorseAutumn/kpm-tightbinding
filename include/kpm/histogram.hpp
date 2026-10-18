#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kpm {

/// Normalized DOS histogram on bins of width dE centered at
/// -4, -4+dE, ..., 4. rho = count / (N * dE), so the histogram
/// integrates to 1. Throws if a sample falls outside [-4-dE/2, 4+dE/2].
inline std::vector<std::pair<double, double>>
dos_histogram(const std::vector<double>& evals, double dE) {
  const int n_bins = static_cast<int>(std::llround(8.0 / dE)) + 1;
  std::vector<std::pair<double, double>> hist(n_bins);
  for (int k = 0; k < n_bins; ++k) {
    hist[k].first = -4.0 + k * dE;
    hist[k].second = 0.0;
  }
  for (double E : evals) {
    const int k = static_cast<int>(std::floor((E + 4.0 + 0.5 * dE) / dE));
    if (k < 0 || k >= n_bins)
      throw std::out_of_range("dos_histogram: eigenvalue outside [-4, 4]");
    hist[k].second += 1.0;
  }
  for (auto& bin : hist)
    bin.second /= static_cast<double>(evals.size()) * dE;
  return hist;
}

} // namespace kpm
