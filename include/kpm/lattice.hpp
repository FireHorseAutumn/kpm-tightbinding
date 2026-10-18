#pragma once

#include <vector>

namespace kpm {

/// CSR sparse matrix, FP64.
struct CsrMatrix {
  int n = 0;                  // matrix dimension
  std::vector<int> row_ptr;   // size n + 1
  std::vector<int> col_idx;   // size nnz
  std::vector<double> values; // size nnz

  int nnz() const { return static_cast<int>(values.size()); }
};

/// LxL square-lattice tight-binding Hamiltonian with open boundary
/// conditions, hopping t = -1, zero on-site energy. Row-major site index
/// (i,j) -> i*L + j. At most 5 entries per row (KPM.md, Model section).
CsrMatrix square_lattice(int L);

/// y = A x.
void spmv(const CsrMatrix& A, const std::vector<double>& x,
          std::vector<double>& y);

} // namespace kpm
