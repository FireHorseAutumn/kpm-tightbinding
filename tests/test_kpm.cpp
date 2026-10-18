// Acceptance tests for the R1 and R2 specifications in docs/KPM.md.
// Every check prints its measured value; any failure exits nonzero.

#include "kpm/exact_oracle.hpp"
#include "kpm/histogram.hpp"
#include "kpm/kpm.hpp"
#include "kpm/lattice.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

constexpr double kPi = std::numbers::pi;
int failures = 0;

void check(bool ok, const char* name, const std::string& detail) {
  std::printf("[%s] %s: %s\n", ok ? "pass" : "FAIL", name, detail.c_str());
  if (!ok)
    ++failures;
}

double max_symmetry_residual(const std::vector<double>& ev) {
  double worst = 0.0;
  const int n = static_cast<int>(ev.size());
  for (int k = 0; k < n; ++k)
    worst = std::max(worst, std::abs(ev[k] + ev[n - 1 - k]));
  return worst;
}

// Mean rho of the fine KPM grid inside the histogram bin centered at
// bin_center (midpoint rule; ~25 grid points per 0.1 bin).
double kpm_bin_average(const std::vector<std::pair<double, double>>& dos,
                       double bin_center, double dE) {
  const double lo = bin_center - 0.5 * dE, hi = bin_center + 0.5 * dE;
  double acc = 0.0;
  int cnt = 0;
  for (const auto& [E, rho] : dos) {
    if (E >= lo && E < hi) {
      acc += rho;
      ++cnt;
    }
  }
  if (cnt == 0)
    throw std::runtime_error("kpm_bin_average: empty bin");
  return acc / cnt;
}

// Trapezoid weight of the KPM DOS inside |E| <= half_window.
double kpm_window_weight(const std::vector<std::pair<double, double>>& dos,
                         double half_window) {
  double acc = 0.0;
  for (size_t k = 1; k < dos.size(); ++k) {
    if (std::abs(dos[k].first) <= half_window &&
        std::abs(dos[k - 1].first) <= half_window)
      acc += 0.5 * (dos[k].second + dos[k - 1].second) *
             (dos[k].first - dos[k - 1].first);
  }
  return acc;
}

double linf_diff(const std::vector<double>& x, const std::vector<double>& y) {
  double worst = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    worst = std::max(worst, std::abs(x[i] - y[i]));
  return worst;
}

double l2_diff(const std::vector<double>& x, const std::vector<double>& y) {
  double acc = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double d = x[i] - y[i];
    acc += d * d;
  }
  return std::sqrt(acc);
}

} // namespace

int main() {
  char buf[256];

  // ---------- R1: exact diagonalization ----------
  const auto ev8 = kpm::exact_eigenvalues(kpm::square_lattice(8));

  double emax = 0.0;
  for (double E : ev8)
    emax = std::max(emax, std::abs(E));
  std::snprintf(buf, sizeof buf, "max|E| = %.15f (bound 4)", emax);
  check(emax <= 4.0 + 1e-12, "R1 L=8 spectrum inside [-4, 4]", buf);

  double trace = 0.0;
  for (double E : ev8)
    trace += E;
  std::snprintf(buf, sizeof buf, "|sum E| = %.3e", std::abs(trace));
  check(std::abs(trace) < 1e-12, "R1 L=8 Tr H = 0", buf);

  const double sym8 = max_symmetry_residual(ev8);
  std::snprintf(buf, sizeof buf, "max|E_k + E_{N-1-k}| = %.3e", sym8);
  check(sym8 < 1e-12, "R1 L=8 spectrum symmetric about 0", buf);

  const auto ev14 = kpm::exact_eigenvalues(kpm::square_lattice(14));
  const double emin_exact = -4.0 * std::cos(kPi / 15.0);
  const double emin_err = std::abs(ev14.front() - emin_exact);
  std::snprintf(buf, sizeof buf, "E_min = %.15f vs %.15f, err %.3e",
                ev14.front(), emin_exact, emin_err);
  check(emin_err < 1e-12, "R1 E_min(L=14) = -4cos(pi/15)", buf);

  const double sym14 = max_symmetry_residual(ev14);
  std::snprintf(buf, sizeof buf, "max|E_k + E_{N-1-k}| = %.3e", sym14);
  check(sym14 < 1e-12, "R1 L=14 spectrum symmetric about 0", buf);

  // ---------- R2: KPM ----------
  const int L = 14;
  const int N = L * L;
  const kpm::CsrMatrix H = kpm::square_lattice(L);
  kpm::KpmParams p; // n_cheb=1000, R=32, a=4.04, b=0, fixed seed
  const auto mu = kpm::chebyshev_moments(H, p);
  const auto g = kpm::jackson_kernel(p.n_cheb);
  const auto dos = kpm::kpm_dos(mu, g, p.a, p.b, 2048);

  // mu_2 has an exact analytic value: T_2(x) = 2x^2 - 1 and
  // Tr H^2 = 4*L*(L-1) t^2, so mu_2 = 2*4(L-1)/(L a^2) - 1. The tolerance
  // is ~4x the stochastic sigma (~0.009) at R=32.
  const double mu2_exact = 2.0 * 4.0 * (L - 1) / (L * p.a * p.a) - 1.0;
  const double mu2_err = std::abs(mu[2] - mu2_exact);
  std::snprintf(buf, sizeof buf, "mu_2 = %.6f vs exact %.6f, err %.3e",
                mu[2], mu2_exact, mu2_err);
  check(mu2_err < 0.04, "R2 mu_2 matches analytic 2*Tr H'^2/N - 1", buf);

  const double integral = kpm::integrate_dos(dos);
  std::snprintf(buf, sizeof buf, "|int rho dE - 1| = %.3e",
                std::abs(integral - 1.0));
  check(std::abs(integral - 1.0) < 1e-3, "R2 int rho dE = 1", buf);

  // Symmetry: grid is symmetric about E=0, so sample k pairs with M-1-k.
  double sym_mean = 0.0, sym_max = 0.0;
  const int M = static_cast<int>(dos.size());
  for (int k = 0; k < M; ++k) {
    const double d = std::abs(dos[k].second - dos[M - 1 - k].second);
    sym_mean += d;
    sym_max = std::max(sym_max, d);
  }
  sym_mean /= M;
  std::snprintf(buf, sizeof buf, "mean|rho(E)-rho(-E)| = %.4f, max %.4f",
                sym_mean, sym_max);
  check(sym_mean < 0.05, "R2 DOS symmetric within stochastic noise", buf);

  // KPM vs R1 histogram on matched bins (dE = 0.1). With only N = 196
  // eigenvalues the histogram's own multinomial sampling noise dominates
  // the L1 distance; measured L1 = 0.1953 with the fixed default seed.
  // The bound below (~1.8x measured) is a regression guard, not an
  // accuracy certificate.
  const double dE = 0.1;
  const auto hist = kpm::dos_histogram(ev14, dE);
  double l1 = 0.0;
  for (const auto& [bin_center, rho_hist] : hist)
    l1 += std::abs(kpm_bin_average(dos, bin_center, dE) - rho_hist) * dE;
  std::snprintf(buf, sizeof buf, "L1 = %.4f (dE = 0.1, N = %d)", l1, N);
  check(l1 < 0.35, "R2 KPM vs R1 histogram L1 distance", buf);

  // Van Hove region: the exact DOS log-diverges at E = 0 and the KPM peak
  // is finite by Jackson construction, so compare integrated weight in a
  // window around 0, never the point value.
  const double half_window = 0.2;
  int window_count = 0;
  for (double E : ev14)
    if (std::abs(E) <= half_window)
      ++window_count;
  const double w_hist = static_cast<double>(window_count) / N;
  const double w_kpm = kpm_window_weight(dos, half_window);
  std::snprintf(buf, sizeof buf,
                "weight(|E|<0.2): KPM %.4f vs histogram %.4f, diff %.4f",
                w_kpm, w_hist, std::abs(w_kpm - w_hist));
  check(std::abs(w_kpm - w_hist) < 0.02,
        "R2 Van Hove window weight agrees with R1", buf);

  // ---------- R2c: moment convergence vs R ----------
  kpm::KpmParams p4 = p, p128 = p;
  p4.n_random = 4;
  p4.seed = p.seed + 4;
  p128.n_random = 128;
  p128.seed = p.seed + 128;
  const auto mu4 = kpm::chebyshev_moments(H, p4);
  const auto mu128 = kpm::chebyshev_moments(H, p128);
  const double d4_inf = linf_diff(mu4, mu128);
  const double d32_inf = linf_diff(mu, mu128);
  const double d4_l2 = l2_diff(mu4, mu128);
  const double d32_l2 = l2_diff(mu, mu128);
  std::printf(
      "[info] moment spread vs R=128: R=4 linf %.4f l2 %.4f | R=32 linf %.4f "
      "l2 %.4f (stochastic sigma ~ 1/sqrt(2*N*R) = %.4f at R=32)\n",
      d4_inf, d4_l2, d32_inf, d32_l2, 1.0 / std::sqrt(2.0 * N * 32));
  std::snprintf(buf, sizeof buf, "|mu(R=32) - mu(R=128)|_inf = %.4f", d32_inf);
  check(d32_inf < 0.06, "R2c R=32 moments stable vs R=128", buf);

  std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES",
              failures, failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
