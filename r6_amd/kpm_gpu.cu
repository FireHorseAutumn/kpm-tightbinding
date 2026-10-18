#include "hip/hip_runtime.h"
// R3 (naive) and R4 (fused) GPU KPM moment kernels. Hand-written CUDA;
// no vendor library kernels anywhere in the measured path.
//
// Analytic global-memory traffic per Chebyshev moment per random vector
// (algorithmic minimum; small lattices may be served from L2 in practice).
// Let rb = sizeof(Real) (8 FP64 / 4 FP32), n = N sites, nnz ~ 4n for the
// square lattice.
//
//   matrix (both modes): nnz*(rb + 4) + 4*(n+1)         [values+col, row_ptr]
//
//   R3 naive per moment:
//     SpMV       : read t_n (rb*n) + matrix, write w (rb*n)
//     recurrence : read w + t_{n-1} (2*rb*n), write t_{n+1} (rb*n)
//     <r|t> dot  : read r + t_{n+1} (2*rb*n)
//     total      = matrix + 7*rb*n      (FP64: matrix + 56n)
//
//   R4 fused per moment:
//     fused      : read t_n + t_{n-1} (2*rb*n) + matrix, write t_{n+1} (rb*n)
//     <r|t> dot  : read r + t_{n+1} (2*rb*n)
//     total      = matrix + 5*rb*n      (FP64: matrix + 40n)
//
// The w round-trip (write in SpMV + read in recurrence, 2*rb*n) vanishes in
// the fused kernel. Vector-traffic ratio naive/fused = 7/5 = 1.40; with the
// matrix included (nnz = 4n exactly, FP64) it is 108n/92n = 1.17. The dot is
// identical work in both modes and is included in the timing and byte count.

#include "kpm_gpu.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define KPM_CUDA_CHECK(call)                                                    \
  do {                                                                          \
    const hipError_t kpm_err_ = (call);                                        \
    if (kpm_err_ != hipSuccess)                                                \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + " in " #call ": " +   \
                               hipGetErrorString(kpm_err_));                   \
  } while (0)

namespace {

constexpr int kBlock = 256;

// w = (A x - b x) / a, one thread per row, CSR. Division by a (not an
// inv_a multiply) so FP64 state vectors stay bitwise identical to the CPU
// oracle path, which computes (y - b*x)/a with the same per-row
// accumulation order.
template <typename Real>
__global__ void spmv_scaled_kernel(const int n,
                                   const int* __restrict__ row_ptr,
                                   const int* __restrict__ col_idx,
                                   const Real* __restrict__ vals, const Real a,
                                   const Real b, const Real* __restrict__ x,
                                   Real* __restrict__ w) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n)
    return;
  Real acc = 0;
  for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p)
    acc += vals[p] * x[col_idx[p]];
  w[row] = (acc - b * x[row]) / a;
}

// R3 recurrence kernel: t_next = 2 w - t_prev.
template <typename Real>
__global__ void recurrence_kernel(const int n, const Real* __restrict__ w,
                                  const Real* __restrict__ t_prev,
                                  Real* __restrict__ t_next) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  t_next[i] = static_cast<Real>(2) * w[i] - t_prev[i];
}

// R4 fused kernel: the row's dot product and the recurrence in one pass;
// w lives in a register and never touches global memory.
template <typename Real>
__global__ void fused_moment_kernel(const int n,
                                    const int* __restrict__ row_ptr,
                                    const int* __restrict__ col_idx,
                                    const Real* __restrict__ vals,
                                    const Real a, const Real b,
                                    const Real* __restrict__ x,
                                    const Real* __restrict__ t_prev,
                                    Real* __restrict__ t_next) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n)
    return;
  Real acc = 0;
  for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p)
    acc += vals[p] * x[col_idx[p]];
  const Real w = (acc - b * x[row]) / a;
  t_next[row] = static_cast<Real>(2) * w - t_prev[row];
}

// <u|v> partial block reduction; one atomicAdd per block into `out`.
// Accumulates in double even for Real = float: one scalar per moment,
// negligible cost, keeps the stochastic-trace reduction from adding FP32
// error on top of the state-vector error being measured.
template <typename Real>
__global__ void dot_kernel(const int n, const Real* __restrict__ u,
                           const Real* __restrict__ v,
                           double* __restrict__ out) {
  __shared__ double sdot[kBlock];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdot[threadIdx.x] =
      (i < n) ? static_cast<double>(u[i]) * static_cast<double>(v[i]) : 0.0;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdot[threadIdx.x] += sdot[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(out, sdot[0]);
}

__global__ void triad_kernel(const size_t n, const double* __restrict__ a,
                             const double* __restrict__ b,
                             double* __restrict__ c, const double s) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + s * b[i];
}

// Owns all device state for one matrix + one slab of R random vectors;
// the moment loop can then be re-run (warmup + timed reps) without any
// further host<->device traffic.
template <typename Real>
class MomentRunner {
public:
  MomentRunner(const kpm::CsrMatrix& H, double a, double b, int n_cheb,
               int n_random, std::uint64_t seed)
      : n_(H.n), nnz_(H.nnz()), n_cheb_(n_cheb), n_random_(n_random),
        a_(static_cast<Real>(a)), b_(static_cast<Real>(b)) {
    // Random +/-1 slab, identical mt19937_64 sequence as the CPU path.
    std::mt19937_64 rng(seed);
    std::vector<Real> slab(static_cast<size_t>(n_) * n_random_);
    for (int r = 0; r < n_random_; ++r)
      for (int i = 0; i < n_; ++i)
        slab[static_cast<size_t>(r) * n_ + i] =
            (rng() & 1ULL) ? static_cast<Real>(1) : static_cast<Real>(-1);

    std::vector<Real> vals(H.values.begin(), H.values.end());

    KPM_CUDA_CHECK(hipMalloc(&row_ptr_, sizeof(int) * (n_ + 1)));
    KPM_CUDA_CHECK(hipMalloc(&col_idx_, sizeof(int) * nnz_));
    KPM_CUDA_CHECK(hipMalloc(&vals_, sizeof(Real) * nnz_));
    KPM_CUDA_CHECK(hipMalloc(&r_slab_, sizeof(Real) * slab.size()));
    KPM_CUDA_CHECK(hipMalloc(&t_prev_, sizeof(Real) * n_));
    KPM_CUDA_CHECK(hipMalloc(&t_cur_, sizeof(Real) * n_));
    KPM_CUDA_CHECK(hipMalloc(&t_next_, sizeof(Real) * n_));
    KPM_CUDA_CHECK(hipMalloc(&w_, sizeof(Real) * n_));
    KPM_CUDA_CHECK(hipMalloc(&mu_dev_, sizeof(double) * (n_cheb_ + 1)));
    KPM_CUDA_CHECK(hipMemcpy(row_ptr_, H.row_ptr.data(),
                              sizeof(int) * (n_ + 1), hipMemcpyHostToDevice));
    KPM_CUDA_CHECK(hipMemcpy(col_idx_, H.col_idx.data(), sizeof(int) * nnz_,
                              hipMemcpyHostToDevice));
    KPM_CUDA_CHECK(hipMemcpy(vals_, vals.data(), sizeof(Real) * nnz_,
                              hipMemcpyHostToDevice));
    KPM_CUDA_CHECK(hipMemcpy(r_slab_, slab.data(), sizeof(Real) * slab.size(),
                              hipMemcpyHostToDevice));
    KPM_CUDA_CHECK(hipDeviceSynchronize());
    KPM_CUDA_CHECK(hipEventCreate(&ev0_));
    KPM_CUDA_CHECK(hipEventCreate(&ev1_));
  }

  ~MomentRunner() {
    hipFree(row_ptr_);
    hipFree(col_idx_);
    hipFree(vals_);
    hipFree(r_slab_);
    hipFree(t_prev_);
    hipFree(t_cur_);
    hipFree(t_next_);
    hipFree(w_);
    hipFree(mu_dev_);
    hipEventDestroy(ev0_);
    hipEventDestroy(ev1_);
  }

  MomentRunner(const MomentRunner&) = delete;
  MomentRunner& operator=(const MomentRunner&) = delete;

  kpm::GpuMomentResult run(kpm::GpuKernelMode mode) {
    const int grid = (n_ + kBlock - 1) / kBlock;
    KPM_CUDA_CHECK(hipMemset(mu_dev_, 0, sizeof(double) * (n_cheb_ + 1)));
    KPM_CUDA_CHECK(hipDeviceSynchronize());
    KPM_CUDA_CHECK(hipEventRecord(ev0_));
    for (int r = 0; r < n_random_; ++r) {
      const Real* rvec = r_slab_ + static_cast<size_t>(r) * n_;
      KPM_CUDA_CHECK(hipDeviceSynchronize());
      KPM_CUDA_CHECK(hipMemcpy(t_prev_, rvec, sizeof(Real) * n_,
                                hipMemcpyDeviceToDevice));
      KPM_CUDA_CHECK(hipDeviceSynchronize());
      // t_0 = r, mu_0 = <r|t_0>
      dot_kernel<<<grid, kBlock>>>(n_, rvec, t_prev_, mu_dev_);
      KPM_CUDA_CHECK(hipGetLastError());
      // t_1 = H' t_0, mu_1 = <r|t_1>
      spmv_scaled_kernel<<<grid, kBlock>>>(n_, row_ptr_, col_idx_, vals_, a_,
                                           b_, t_prev_, t_cur_);
      KPM_CUDA_CHECK(hipGetLastError());
      dot_kernel<<<grid, kBlock>>>(n_, rvec, t_cur_, mu_dev_ + 1);
      KPM_CUDA_CHECK(hipGetLastError());
      // t_{k+1} = 2 H' t_k - t_{k-1}
      for (int k = 1; k < n_cheb_; ++k) {
        if (mode == kpm::GpuKernelMode::Naive) {
          spmv_scaled_kernel<<<grid, kBlock>>>(n_, row_ptr_, col_idx_, vals_,
                                               a_, b_, t_cur_, w_);
          KPM_CUDA_CHECK(hipGetLastError());
          recurrence_kernel<<<grid, kBlock>>>(n_, w_, t_prev_, t_next_);
          KPM_CUDA_CHECK(hipGetLastError());
        } else {
          fused_moment_kernel<<<grid, kBlock>>>(n_, row_ptr_, col_idx_, vals_,
                                                a_, b_, t_cur_, t_prev_,
                                                t_next_);
          KPM_CUDA_CHECK(hipGetLastError());
        }
        dot_kernel<<<grid, kBlock>>>(n_, rvec, t_next_, mu_dev_ + k + 1);
        KPM_CUDA_CHECK(hipGetLastError());
        std::swap(t_prev_, t_cur_);
        std::swap(t_cur_, t_next_);
      }
    }
    KPM_CUDA_CHECK(hipEventRecord(ev1_));
    KPM_CUDA_CHECK(hipEventSynchronize(ev1_));
    KPM_CUDA_CHECK(hipDeviceSynchronize());
    float ms = 0.0f;
    KPM_CUDA_CHECK(hipEventElapsedTime(&ms, ev0_, ev1_));

    kpm::GpuMomentResult res;
    res.moments.resize(n_cheb_ + 1);
    KPM_CUDA_CHECK(hipMemcpy(res.moments.data(), mu_dev_,
                              sizeof(double) * (n_cheb_ + 1),
                              hipMemcpyDeviceToHost));
    const double scale =
        1.0 / (static_cast<double>(n_) * static_cast<double>(n_random_));
    for (double& m : res.moments)
      m *= scale;
    res.seconds = static_cast<double>(ms) * 1e-3;
    return res;
  }

private:
  int n_, nnz_, n_cheb_, n_random_;
  Real a_, b_;
  int* row_ptr_ = nullptr;
  int* col_idx_ = nullptr;
  Real* vals_ = nullptr;
  Real* r_slab_ = nullptr;
  Real* t_prev_ = nullptr;
  Real* t_cur_ = nullptr;
  Real* t_next_ = nullptr;
  Real* w_ = nullptr;
  double* mu_dev_ = nullptr;
  hipEvent_t ev0_{}, ev1_{};
};

template <typename Real>
kpm::GpuMomentResult moments_impl(const kpm::CsrMatrix& H,
                                  const kpm::KpmParams& p,
                                  kpm::GpuKernelMode mode) {
  MomentRunner<Real> runner(H, p.a, p.b, p.n_cheb, p.n_random, p.seed);
  return runner.run(mode);
}

template <typename Real>
kpm::GpuBenchResult bench_impl(const kpm::CsrMatrix& H,
                               const kpm::KpmParams& p, kpm::GpuKernelMode mode,
                               int n_reps) {
  MomentRunner<Real> runner(H, p.a, p.b, p.n_cheb, p.n_random, p.seed);
  runner.run(mode); // warmup, untimed (result discarded)
  std::vector<double> times;
  times.reserve(n_reps);
  for (int rep = 0; rep < n_reps; ++rep)
    times.push_back(runner.run(mode).seconds);
  std::sort(times.begin(), times.end());
  kpm::GpuBenchResult res;
  res.median_s = times[times.size() / 2];
  res.min_s = times.front();
  res.max_s = times.back();
  res.n_reps = n_reps;
  return res;
}

} // namespace

namespace kpm {

GpuMomentResult chebyshev_moments_gpu(const CsrMatrix& H, const KpmParams& p,
                                      GpuKernelMode mode) {
  return moments_impl<double>(H, p, mode);
}

GpuMomentResult chebyshev_moments_gpu_fp32(const CsrMatrix& H,
                                           const KpmParams& p,
                                           GpuKernelMode mode) {
  return moments_impl<float>(H, p, mode);
}

GpuBenchResult bench_moment_loop_fp64(const CsrMatrix& H, const KpmParams& p,
                                      GpuKernelMode mode, int n_reps) {
  return bench_impl<double>(H, p, mode, n_reps);
}

GpuBenchResult bench_moment_loop_fp32(const CsrMatrix& H, const KpmParams& p,
                                      GpuKernelMode mode, int n_reps) {
  return bench_impl<float>(H, p, mode, n_reps);
}

double bench_triad_fp64(double& min_gbps, double& max_gbps, int n_reps) {
  const size_t n = size_t{64} << 20; // 64M doubles = 512 MiB per array
  double *a = nullptr, *b = nullptr, *c = nullptr;
  KPM_CUDA_CHECK(hipMalloc(&a, sizeof(double) * n));
  KPM_CUDA_CHECK(hipMalloc(&b, sizeof(double) * n));
  KPM_CUDA_CHECK(hipMalloc(&c, sizeof(double) * n));
  KPM_CUDA_CHECK(hipMemset(a, 0, sizeof(double) * n));
  KPM_CUDA_CHECK(hipMemset(b, 0, sizeof(double) * n));
  const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  hipEvent_t ev0, ev1;
  KPM_CUDA_CHECK(hipEventCreate(&ev0));
  KPM_CUDA_CHECK(hipEventCreate(&ev1));

  std::vector<double> gbps;
  gbps.reserve(n_reps);
  const double bytes = 3.0 * sizeof(double) * n;
  for (int rep = 0; rep < n_reps + 1; ++rep) { // first rep is warmup
    KPM_CUDA_CHECK(hipEventRecord(ev0));
    triad_kernel<<<grid, kBlock>>>(n, a, b, c, 3.0);
    KPM_CUDA_CHECK(hipGetLastError());
    KPM_CUDA_CHECK(hipEventRecord(ev1));
    KPM_CUDA_CHECK(hipEventSynchronize(ev1));
    float ms = 0.0f;
    KPM_CUDA_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    if (rep > 0)
      gbps.push_back(bytes / (static_cast<double>(ms) * 1e-3) / 1e9);
  }
  hipEventDestroy(ev0);
  hipEventDestroy(ev1);
  hipFree(a);
  hipFree(b);
  hipFree(c);

  const auto [lo, hi] = std::minmax_element(gbps.begin(), gbps.end());
  min_gbps = *lo;
  max_gbps = *hi;
  std::sort(gbps.begin(), gbps.end());
  return gbps[gbps.size() / 2];
}

double naive_bytes_per_moment(int nnz, int n, int real_bytes) {
  const double matrix =
      static_cast<double>(nnz) * (real_bytes + 4) + 4.0 * (n + 1);
  return matrix + 7.0 * real_bytes * n;
}

double fused_bytes_per_moment(int nnz, int n, int real_bytes) {
  const double matrix =
      static_cast<double>(nnz) * (real_bytes + 4) + 4.0 * (n + 1);
  return matrix + 5.0 * real_bytes * n;
}

std::string gpu_device_name() {
  int dev = 0;
  KPM_CUDA_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop{};
  KPM_CUDA_CHECK(hipGetDeviceProperties(&prop, dev));
  return std::string(prop.name) + " (sm_" + std::to_string(prop.major) +
         std::to_string(prop.minor) + ", " +
         std::to_string(prop.multiProcessorCount) + " SMs, L2 " +
         std::to_string(prop.l2CacheSize / (1024 * 1024)) + " MiB)";
}

} // namespace kpm
