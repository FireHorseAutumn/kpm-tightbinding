#pragma once

#include "kpm/kpm.hpp"
#include "kpm/lattice.hpp"

#include <string>
#include <vector>

namespace kpm {

/// Which GPU moment path to run (docs/KPM.md sections R3 and R4).
enum class GpuKernelMode {
  Naive, // R3: separate CSR SpMV and recurrence kernels per moment; the
         // SpMV output w round-trips through global memory every moment.
  Fused, // R4: SpMV + recurrence in one kernel; w lives in a register and
         // never touches global memory.
};

/// Result of a full GPU moment run.
struct GpuMomentResult {
  std::vector<double> moments; // mu_0..mu_{n_cheb}
  double seconds = 0.0;        // device time of the whole moment loop
};

/// FP64 stochastic Chebyshev moments on the GPU. Random +/-1 vectors are
/// drawn on the host with the identical mt19937_64 sequence as the CPU
/// path, so for the same seed the r vectors are bitwise identical; the
/// rescaled SpMV uses division by a (not an inv_a multiply) so the FP64
/// state vectors match the CPU path bitwise as well. Only the <r|t_n>
/// reduction order differs (tree reduction + atomicAdd vs sequential).
GpuMomentResult chebyshev_moments_gpu(const CsrMatrix& H, const KpmParams& p,
                                      GpuKernelMode mode);

/// FP32 variant: CSR values and vectors are downcast to float once at
/// upload; the SpMV/recurrence arithmetic is FP32. The per-moment <r|t>
/// reduction accumulates in double (one scalar per moment; negligible
/// cost). FP32 is included for the precision/performance comparison;
/// FP64 remains the default for physics runs.
GpuMomentResult chebyshev_moments_gpu_fp32(const CsrMatrix& H,
                                           const KpmParams& p,
                                           GpuKernelMode mode);

/// Timing statistics over repeated runs of the full moment loop.
struct GpuBenchResult {
  double median_s = 0.0;
  double min_s = 0.0;
  double max_s = 0.0;
  int n_reps = 0;
  std::vector<double> reps_s; // per-rep device seconds, run order
};

/// Timed repetitions of the full moment loop (R vectors x n_cheb moments,
/// per-moment <r|t> reductions included — identical work for both modes).
/// The matrix and the random-vector slab are uploaded once; one warmup
/// rep is run and excluded. Every kernel launch is error-checked.
GpuBenchResult bench_moment_loop_fp64(const CsrMatrix& H, const KpmParams& p,
                                      GpuKernelMode mode, int n_reps);
GpuBenchResult bench_moment_loop_fp32(const CsrMatrix& H, const KpmParams& p,
                                      GpuKernelMode mode, int n_reps);

/// STREAM-like triad result: c = a + s*b in FP64 with arrays far larger
/// than L2 (512 MiB each). This is the roofline reference — the spec sheet
/// number is never used.
struct TriadBenchResult {
  double median_gbps = 0.0;
  double min_gbps = 0.0;
  double max_gbps = 0.0;
  int n_reps = 0;
  std::vector<double> reps_gbps; // per-rep effective GB/s, run order
};

TriadBenchResult bench_triad_fp64(int n_reps);

/// Analytic global-memory traffic per Chebyshev moment per random vector,
/// in bytes (algorithmic minimum; small lattices may be served from L2 in
/// practice). real_bytes = 8 (FP64) or 4 (FP32). Derivation in kpm_gpu.cu.
double naive_bytes_per_moment(int nnz, int n, int real_bytes);
double fused_bytes_per_moment(int nnz, int n, int real_bytes);

/// Device properties plus driver/runtime versions, for benchmark metadata.
/// Clock rates are the device-reported maxima, not live measurements.
struct GpuDeviceInfo {
  std::string name;
  int major = 0;            // compute capability
  int minor = 0;
  int sm_count = 0;
  int clock_rate_khz = 0;        // prop.clockRate (max graphics clock)
  int memory_clock_rate_khz = 0; // prop.memoryClockRate
  int memory_bus_width_bits = 0;
  int l2_bytes = 0;
  std::uint64_t global_mem_bytes = 0;
  int driver_version = 0;  // cudaDriverGetVersion, e.g. 13020
  int runtime_version = 0; // cudaRuntimeGetVersion
};

GpuDeviceInfo gpu_device_info();

/// CUDA device name, for report headers.
std::string gpu_device_name();

/// Build toolchain of the GPU library: nvcc version, host compiler,
/// C++ standard, build type.
std::string gpu_toolchain();

} // namespace kpm
