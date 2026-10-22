#include "kpm/kpm.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>

namespace kpm {

namespace {

constexpr double kPi = std::numbers::pi;

// y = (A x - b x) / a, the rescaled Hamiltonian applied to x.
void spmv_scaled(const CsrMatrix& A, double a, double b,
                 const std::vector<double>& x, std::vector<double>& y) {
  for (int row = 0; row < A.n; ++row) {
    double acc = 0.0;
    for (int p = A.row_ptr[row]; p < A.row_ptr[row + 1]; ++p)
      acc += A.values[p] * x[A.col_idx[p]];
    y[row] = (acc - b * x[row]) / a;
  }
}

double dot(const std::vector<double>& u, const std::vector<double>& v) {
  double acc = 0.0;
  for (size_t i = 0; i < u.size(); ++i)
    acc += u[i] * v[i];
  return acc;
}

} // namespace

std::vector<double> jackson_kernel(int n_cheb) {
  if (n_cheb < 1)
    throw std::invalid_argument("jackson_kernel: n_cheb must be >= 1");
  std::vector<double> g(n_cheb + 1);
  const double denom = static_cast<double>(n_cheb) + 1.0;
  const double cot = 1.0 / std::tan(kPi / denom);
  for (int n = 0; n <= n_cheb; ++n) {
    const double phi = kPi * n / denom;
    g[n] = ((n_cheb - n + 1.0) * std::cos(phi) + std::sin(phi) * cot) / denom;
  }
  return g;
}

std::vector<double> chebyshev_moments(const CsrMatrix& H,
                                      const KpmParams& p) {
  if (p.n_cheb < 1)
    throw std::invalid_argument("chebyshev_moments: n_cheb must be >= 1");
  if (p.n_random < 1)
    throw std::invalid_argument("chebyshev_moments: n_random must be >= 1");
  if (p.a <= 4.0)
    throw std::invalid_argument(
        "chebyshev_moments: a must exceed the spectral bound 4 (eta > 0)");
  const int n = H.n;
  std::vector<double> mu(p.n_cheb + 1, 0.0);
  std::mt19937_64 rng(p.seed);
  std::vector<double> rvec(n), t_prev(n), t_cur(n), t_next(n);
  for (int r = 0; r < p.n_random; ++r) {
    for (int i = 0; i < n; ++i)
      rvec[i] = (rng() & 1ULL) ? 1.0 : -1.0;
    t_prev = rvec;
    mu[0] += dot(rvec, t_prev);
    spmv_scaled(H, p.a, p.b, t_prev, t_cur);
    mu[1] += dot(rvec, t_cur);
    for (int k = 1; k < p.n_cheb; ++k) {
      spmv_scaled(H, p.a, p.b, t_cur, t_next);
      for (int i = 0; i < n; ++i)
        t_next[i] = 2.0 * t_next[i] - t_prev[i];
      mu[k + 1] += dot(rvec, t_next);
      std::swap(t_prev, t_cur);
      std::swap(t_cur, t_next);
    }
  }
  const double scale = 1.0 / (static_cast<double>(n) * p.n_random);
  for (double& m : mu)
    m *= scale;
  return mu;
}

std::vector<std::pair<double, double>>
kpm_dos(const std::vector<double>& moments,
        const std::vector<double>& jackson, double a, double b,
        int n_grid) {
  if (moments.size() < 2 || jackson.size() < moments.size())
    throw std::invalid_argument("kpm_dos: moments/jackson size mismatch");
  if (n_grid < 2)
    throw std::invalid_argument("kpm_dos: n_grid must be >= 2");
  const int n_cheb = static_cast<int>(moments.size()) - 1;
  std::vector<double> w(n_cheb);
  for (int n = 1; n <= n_cheb - 1; ++n)
    w[n] = 2.0 * jackson[n] * moments[n];
  std::vector<std::pair<double, double>> out;
  out.reserve(n_grid);
  for (int k = 0; k < n_grid; ++k) {
    const double theta = kPi * (k + 0.5) / n_grid;
    const double eps = std::cos(theta);
    double sum = jackson[0] * moments[0];
    double t_prev = 1.0, t_cur = eps;
    for (int n = 1; n <= n_cheb - 1; ++n) {
      sum += w[n] * t_cur;
      const double t_next = 2.0 * eps * t_cur - t_prev;
      t_prev = t_cur;
      t_cur = t_next;
    }
    const double rho_prime = sum / (kPi * std::sin(theta));
    out.emplace_back(a * eps + b, rho_prime / a);
  }
  std::sort(out.begin(), out.end());
  return out;
}

double integrate_dos(const std::vector<std::pair<double, double>>& dos) {
  double acc = 0.0;
  for (size_t k = 1; k < dos.size(); ++k)
    acc += 0.5 * (dos[k].second + dos[k - 1].second) *
           (dos[k].first - dos[k - 1].first);
  return acc;
}

} // namespace kpm
