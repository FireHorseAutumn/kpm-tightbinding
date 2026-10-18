#pragma once

#include "kpm/lattice.hpp"

#include <Eigen/Dense>

#include <stdexcept>
#include <vector>

namespace kpm {

/// Dense exact-diagonalization reference. Eigen is used only here, not in
/// the measured KPM path. Returns the eigenvalues of A in ascending order;
/// throws if the eigensolve fails.
inline std::vector<double> exact_eigenvalues(const CsrMatrix& A) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(A.n, A.n);
  for (int row = 0; row < A.n; ++row)
    for (int p = A.row_ptr[row]; p < A.row_ptr[row + 1]; ++p)
      H(row, A.col_idx[p]) = A.values[p];
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error("exact_eigenvalues: eigensolve failed");
  const auto& ev = solver.eigenvalues();
  return std::vector<double>(ev.data(), ev.data() + ev.size());
}

} // namespace kpm
