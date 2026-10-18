#pragma once

#include "kpm/lattice.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace kpm {

/// Parameters for the stochastic KPM expansion (KPM.md section R2).
/// The Hamiltonian is rescaled as H' = (H - b*I)/a so that
/// spec(H') is contained in [-1, 1]; for the square lattice a = 4*(1+eta)
/// with safety margin eta = 0.01.
struct KpmParams {
  int n_cheb = 1000;         // highest Chebyshev moment index computed
  int n_random = 32;         // stochastic trace vectors (random +/-1)
  double a = 4.0 * 1.01;     // rescale denominator
  double b = 0.0;            // rescale shift
  std::uint64_t seed = 0x9E3779B9ULL;
};

/// Jackson damping factors g_n for n = 0..n_cheb (RMP 78, 275 (2006),
/// Eq. 71). g_0 = 1, g_n -> 0 as n -> n_cheb.
std::vector<double> jackson_kernel(int n_cheb);

/// Stochastic Chebyshev moments of the rescaled Hamiltonian:
/// mu_n = (1/N)(1/R) sum_r <r|T_n(H')|r>, via the vector recurrence
/// t_0 = r, t_1 = H' r, t_{n+1} = 2 H' t_n - t_{n-1}.
/// Returns mu_0..mu_{n_cheb} (size n_cheb + 1). Deterministic for a
/// fixed seed.
std::vector<double> chebyshev_moments(const CsrMatrix& H,
                                      const KpmParams& p);

/// DOS reconstruction on an energy grid: n_grid points uniform in
/// theta = arccos(eps), eps in (-1, 1):
///   rho'(eps) = [g_0 mu_0 + 2 sum_{n=1}^{N-1} g_n mu_n T_n(eps)]
///               / (pi sqrt(1 - eps^2))
/// then unscaled: rho(E) = rho'((E - b)/a) / a with E = a*eps + b.
/// T_n is evaluated by the recurrence T_{n+1} = 2 eps T_n - T_{n-1}
/// (never cos(n arccos eps)). Returns (E, rho) pairs sorted ascending
/// in E.
std::vector<std::pair<double, double>>
kpm_dos(const std::vector<double>& moments,
        const std::vector<double>& jackson, double a, double b,
        int n_grid);

/// Trapezoid integral of rho over (E, rho) samples sorted ascending in E.
double integrate_dos(const std::vector<std::pair<double, double>>& dos);

} // namespace kpm
