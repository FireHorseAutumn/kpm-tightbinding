#include "kpm/lattice.hpp"

#include <stdexcept>

namespace kpm {

CsrMatrix square_lattice(int L) {
  if (L < 2)
    throw std::invalid_argument("square_lattice: L must be >= 2");
  CsrMatrix A;
  A.n = L * L;
  A.row_ptr.reserve(A.n + 1);
  A.row_ptr.push_back(0);
  A.col_idx.reserve(A.n * 5);
  A.values.reserve(A.n * 5);
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      const int s = i * L + j;
      if (i > 0) {
        A.col_idx.push_back(s - L);
        A.values.push_back(-1.0);
      }
      if (i < L - 1) {
        A.col_idx.push_back(s + L);
        A.values.push_back(-1.0);
      }
      if (j > 0) {
        A.col_idx.push_back(s - 1);
        A.values.push_back(-1.0);
      }
      if (j < L - 1) {
        A.col_idx.push_back(s + 1);
        A.values.push_back(-1.0);
      }
      A.row_ptr.push_back(static_cast<int>(A.col_idx.size()));
    }
  }
  return A;
}

void spmv(const CsrMatrix& A, const std::vector<double>& x,
          std::vector<double>& y) {
  for (int row = 0; row < A.n; ++row) {
    double acc = 0.0;
    for (int p = A.row_ptr[row]; p < A.row_ptr[row + 1]; ++p)
      acc += A.values[p] * x[A.col_idx[p]];
    y[row] = acc;
  }
}

} // namespace kpm
