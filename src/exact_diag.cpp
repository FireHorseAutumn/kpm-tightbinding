#include "kpm/exact_oracle.hpp"
#include "kpm/histogram.hpp"
#include "kpm/lattice.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

void write_dos_csv(const std::string& path,
                   const std::vector<std::pair<double, double>>& dos) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  f << "E,rho\n";
  f.precision(17);
  for (const auto& [E, rho] : dos)
    f << E << ',' << rho << '\n';
  if (!f)
    throw std::runtime_error("write failed: " + path);
}

} // namespace

int main() {
  const std::string data_dir = KPM_DATA_DIR;
  const double dE = 0.1;
  try {
    for (int L = 8; L <= 14; ++L) {
      const kpm::CsrMatrix H = kpm::square_lattice(L);
      const auto t0 = std::chrono::steady_clock::now();
      const std::vector<double> evals = kpm::exact_eigenvalues(H);
      const auto t1 = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      const double trace =
          std::accumulate(evals.begin(), evals.end(), 0.0);
      std::printf("L=%2d N=%4d nnz=%5d  E_min=%+.15f  sum(E)=%+.3e  "
                  "eigensolve %.1f ms\n",
                  L, H.n, H.nnz(), evals.front(), trace, ms);
      write_dos_csv(data_dir + "/exact_dos_L" + std::to_string(L) + ".csv",
                    kpm::dos_histogram(evals, dE));
      if (L == 14) {
        std::ofstream f(data_dir + "/exact_eigvals_L14.csv");
        if (!f)
          throw std::runtime_error("cannot open eigenvalue output");
        f << "E\n";
        f.precision(17);
        for (double E : evals)
          f << E << '\n';
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "exact_diag failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
