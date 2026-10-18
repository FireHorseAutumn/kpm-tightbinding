// GPU moments must match the CPU KPM result for the same seed and random
// vectors. The FP64 SpMV uses the CPU per-row accumulation order, so only
// the <r|t_n> reduction order differs. FP32 uses a separate tolerance.

#include "kpm/kpm.hpp"
#include "kpm/kpm_gpu.hpp"
#include "kpm/lattice.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* name, const std::string& detail) {
  std::printf("[%s] %s: %s\n", ok ? "pass" : "FAIL", name, detail.c_str());
  if (!ok)
    ++failures;
}

double linf_diff(const std::vector<double>& x, const std::vector<double>& y) {
  double worst = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    worst = std::max(worst, std::abs(x[i] - y[i]));
  return worst;
}

} // namespace

int main() {
  char buf[256];
  try {
    std::printf("device: %s\n", kpm::gpu_device_name().c_str());

    // ---------- L=14, n_cheb=1000, R=32 (matches the R2 CPU run) ----------
    const kpm::CsrMatrix H = kpm::square_lattice(14);
    const kpm::KpmParams p; // defaults: n_cheb=1000, R=32, a=4.04, b=0, seed

    const double matrix64 = H.nnz() * 12.0 + 4.0 * (H.n + 1);
    const double matrix32 = H.nnz() * 8.0 + 4.0 * (H.n + 1);
    const bool traffic_ok =
        kpm::naive_bytes_per_moment(H.nnz(), H.n, 8) == matrix64 + 56.0 * H.n &&
        kpm::fused_bytes_per_moment(H.nnz(), H.n, 8) == matrix64 + 40.0 * H.n &&
        kpm::naive_bytes_per_moment(H.nnz(), H.n, 4) == matrix32 + 28.0 * H.n &&
        kpm::fused_bytes_per_moment(H.nnz(), H.n, 4) == matrix32 + 20.0 * H.n;
    check(traffic_ok, "R3/R4 analytic traffic model coefficients",
          "FP64: naive matrix+56n, fused matrix+40n; FP32: naive matrix+28n, fused matrix+20n");

    const auto mu_cpu = kpm::chebyshev_moments(H, p);

    const auto r64n = kpm::chebyshev_moments_gpu(H, p, kpm::GpuKernelMode::Naive);
    const auto r64f = kpm::chebyshev_moments_gpu(H, p, kpm::GpuKernelMode::Fused);

    const double d64n = linf_diff(r64n.moments, mu_cpu);
    std::snprintf(buf, sizeof buf, "linf |mu_gpu_naive - mu_cpu| = %.3e", d64n);
    check(d64n < 1e-12, "R3 FP64 naive GPU moments match CPU (L=14)", buf);

    const double d64f = linf_diff(r64f.moments, mu_cpu);
    std::snprintf(buf, sizeof buf, "linf |mu_gpu_fused - mu_cpu| = %.3e", d64f);
    check(d64f < 1e-12, "R4 FP64 fused GPU moments match CPU (L=14)", buf);

    const auto r32n =
        kpm::chebyshev_moments_gpu_fp32(H, p, kpm::GpuKernelMode::Naive);
    const auto r32f =
        kpm::chebyshev_moments_gpu_fp32(H, p, kpm::GpuKernelMode::Fused);
    const double d32n = linf_diff(r32n.moments, mu_cpu);
    const double d32f = linf_diff(r32f.moments, mu_cpu);
    std::snprintf(buf, sizeof buf,
                  "linf |mu_fp32 - mu_cpu|: naive %.3e, fused %.3e (loose "
                  "stochastic bound; FP64 is the physics default)",
                  d32n, d32f);
    check(d32n < 1e-3 && d32f < 1e-3,
          "R3/R4 FP32 moments within FP32 noise of CPU FP64", buf);

    // ---------- L=8 edge case: n = 64 < one 256-thread block ----------
    const kpm::CsrMatrix H8 = kpm::square_lattice(8);
    const auto mu8_cpu = kpm::chebyshev_moments(H8, p);
    const auto r8f = kpm::chebyshev_moments_gpu(H8, p, kpm::GpuKernelMode::Fused);
    const double d8 = linf_diff(r8f.moments, mu8_cpu);
    std::snprintf(buf, sizeof buf, "linf |mu_gpu_fused - mu_cpu| = %.3e", d8);
    check(d8 < 1e-12, "R4 FP64 fused GPU moments match CPU (L=8, n < block)",
          buf);

    // ---------- L=17: n = 289 > kBlock, first multi-block grid ----------
    // Exercises the cross-block atomicAdd in the <r|t> reduction, which the
    // L=14/L=8 single-block runs never touch.
    const kpm::CsrMatrix H17 = kpm::square_lattice(17);
    const auto mu17_cpu = kpm::chebyshev_moments(H17, p);
    const auto r17n = kpm::chebyshev_moments_gpu(H17, p, kpm::GpuKernelMode::Naive);
    const auto r17f = kpm::chebyshev_moments_gpu(H17, p, kpm::GpuKernelMode::Fused);

    const double d17n = linf_diff(r17n.moments, mu17_cpu);
    std::snprintf(buf, sizeof buf, "linf |mu_gpu_naive - mu_cpu| = %.3e", d17n);
    check(d17n < 1e-13, "R3 FP64 naive GPU moments match CPU (L=17, multi-block)",
          buf);

    const double d17f = linf_diff(r17f.moments, mu17_cpu);
    std::snprintf(buf, sizeof buf, "linf |mu_gpu_fused - mu_cpu| = %.3e", d17f);
    check(d17f < 1e-13, "R4 FP64 fused GPU moments match CPU (L=17, multi-block)",
          buf);

    // Naive and fused apply identical per-row arithmetic; only the cross-block
    // atomicAdd arrival order in the dot can differ.
    const double d17nf = linf_diff(r17n.moments, r17f.moments);
    std::snprintf(buf, sizeof buf, "linf |mu_naive - mu_fused| = %.3e", d17nf);
    check(d17nf < 1e-14, "R3/R4 GPU moments agree (L=17, multi-block)", buf);

    // ---------- large sizes: benchmark-class grids, reduced moment work ----------
    // CPU oracle stays cheap at these n_cheb/R; the point is the grid path,
    // not statistics. Looser bound than L<=17: the reduction sums more terms
    // (observed ~1e-14 at L=128, ~7e-14 at L=256).
    kpm::KpmParams p128;
    p128.n_cheb = 256;
    p128.n_random = 4;
    const kpm::CsrMatrix H128 = kpm::square_lattice(128);
    const auto mu128_cpu = kpm::chebyshev_moments(H128, p128);
    const auto r128n =
        kpm::chebyshev_moments_gpu(H128, p128, kpm::GpuKernelMode::Naive);
    const auto r128f =
        kpm::chebyshev_moments_gpu(H128, p128, kpm::GpuKernelMode::Fused);
    const double d128n = linf_diff(r128n.moments, mu128_cpu);
    const double d128f = linf_diff(r128f.moments, mu128_cpu);
    std::snprintf(buf, sizeof buf,
                  "linf |mu_gpu - mu_cpu|: naive %.3e, fused %.3e", d128n,
                  d128f);
    check(d128n < 1e-12 && d128f < 1e-12,
          "FP64 GPU moments match CPU (L=128, 64 blocks)", buf);

    kpm::KpmParams p256;
    p256.n_cheb = 64;
    p256.n_random = 2;
    const kpm::CsrMatrix H256 = kpm::square_lattice(256);
    const auto mu256_cpu = kpm::chebyshev_moments(H256, p256);
    const auto r256f =
        kpm::chebyshev_moments_gpu(H256, p256, kpm::GpuKernelMode::Fused);
    const double d256f = linf_diff(r256f.moments, mu256_cpu);
    std::snprintf(buf, sizeof buf, "linf |mu_gpu_fused - mu_cpu| = %.3e",
                  d256f);
    check(d256f < 1e-12, "FP64 fused GPU moments match CPU (L=256, 256 blocks)",
          buf);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_kpm failed: %s\n", e.what());
    return 1;
  }

  std::printf("%s (%d failure%s)\n",
              failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
