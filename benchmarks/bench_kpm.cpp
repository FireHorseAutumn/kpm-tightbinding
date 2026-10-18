// R3 vs R4 benchmark driver: naive (separate SpMV + recurrence kernels)
// vs fused (one kernel per moment), FP64 and FP32, lattice sweep, median
// of N reps with min/max spread, effective GB/s against a measured
// STREAM-like triad bandwidth on the same GPU. JSON out to data/ with raw
// per-rep timings and device/driver/seed/toolchain metadata.
//
// Absolute times vary with machine state; report medians and observed spreads.

#include "kpm/kpm_gpu.hpp"
#include "kpm/lattice.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SizeEntry {
  int L;
  int n;
  int nnz;
  kpm::GpuBenchResult fp64_naive, fp64_fused, fp32_naive, fp32_fused;
};

std::vector<int> parse_sizes(const std::string& s) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t comma = s.find(',', pos);
    out.push_back(std::stoi(s.substr(pos, comma - pos)));
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

// Device seconds per single Chebyshev moment per random vector.
double s_per_moment(const kpm::GpuBenchResult& r, int n_cheb, int n_random) {
  return r.median_s / (static_cast<double>(n_cheb) * n_random);
}

void json_result(std::ofstream& f, const kpm::GpuBenchResult& r, int n_cheb,
                 int n_random, double bytes_per_moment, int indent) {
  const std::string pad(indent, ' ');
  const double spm = s_per_moment(r, n_cheb, n_random);
  f << pad << "{\n"
    << pad << "  \"median_s\": " << r.median_s << ",\n"
    << pad << "  \"min_s\": " << r.min_s << ",\n"
    << pad << "  \"max_s\": " << r.max_s << ",\n"
    << pad << "  \"us_per_moment\": " << spm * 1e6 << ",\n"
    << pad << "  \"bytes_per_moment\": " << bytes_per_moment << ",\n"
    << pad << "  \"effective_gbps\": " << bytes_per_moment / spm / 1e9 << ",\n"
    << pad << "  \"reps_s\": [";
  for (size_t i = 0; i < r.reps_s.size(); ++i)
    f << (i ? ", " : "") << r.reps_s[i];
  f << "]\n"
    << pad << "}";
}

std::string timestamp_utc() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

std::string hex_seed(std::uint64_t seed) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "0x%llx",
                static_cast<unsigned long long>(seed));
  return buf;
}

} // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes = {16, 32, 64, 128};
  kpm::KpmParams p;
  p.n_random = 4; // timing only; the R factor is identical work per moment
  int n_reps = 11;
  int triad_reps = 20;
  std::string out_path = std::string(KPM_DATA_DIR) + "/kpm_gpu_bench.json";
  for (int i = 1; i < argc; ++i) {
    try {
      if (std::strncmp(argv[i], "--sizes=", 8) == 0)
        sizes = parse_sizes(argv[i] + 8);
      else if (std::strncmp(argv[i], "--n-cheb=", 9) == 0)
        p.n_cheb = std::stoi(argv[i] + 9);
      else if (std::strncmp(argv[i], "--R=", 4) == 0)
        p.n_random = std::stoi(argv[i] + 4);
      else if (std::strncmp(argv[i], "--reps=", 7) == 0)
        n_reps = std::stoi(argv[i] + 7);
      else if (std::strncmp(argv[i], "--triad-reps=", 13) == 0)
        triad_reps = std::stoi(argv[i] + 13);
      else if (std::strncmp(argv[i], "--out=", 6) == 0)
        out_path = argv[i] + 6;
      else
        throw std::invalid_argument(std::string("unknown argument: ") +
                                    argv[i]);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "bench_kpm: %s\n", e.what());
      std::fprintf(stderr,
                   "usage: bench_kpm [--sizes=16,32,64,128] [--n-cheb=1000] "
                   "[--R=4] [--reps=11] [--triad-reps=20] [--out=PATH]\n");
      return 1;
    }
  }

  try {
    const kpm::GpuDeviceInfo devinfo = kpm::gpu_device_info();
    const std::string device = kpm::gpu_device_name();
    std::printf("device: %s\n", device.c_str());

    const kpm::TriadBenchResult triad = kpm::bench_triad_fp64(triad_reps);
    std::printf("STREAM-like triad FP64 (3 x 512 MiB arrays): median %.1f "
                "GB/s (min %.1f, max %.1f, %d reps)\n\n",
                triad.median_gbps, triad.min_gbps, triad.max_gbps,
                triad.n_reps);

    std::printf("moment loop: n_cheb=%d, R=%d, median of %d reps (device "
                "time, per-moment <r|t> reductions included)\n\n",
                p.n_cheb, p.n_random, n_reps);
    std::printf("%4s %8s | %9s %9s | %9s %9s | %6s | %7s %7s | %6s %6s\n",
                "L", "N", "naiv64", "fuse64", "naiv32", "fuse32", "spd64",
                "GB/s64n", "GB/s64f", "%BW64n", "%BW64f");

    std::vector<SizeEntry> entries;
    for (const int L : sizes) {
      const kpm::CsrMatrix H = kpm::square_lattice(L);
      SizeEntry e{L, H.n, H.nnz(),
                  kpm::bench_moment_loop_fp64(H, p, kpm::GpuKernelMode::Naive,
                                              n_reps),
                  kpm::bench_moment_loop_fp64(H, p, kpm::GpuKernelMode::Fused,
                                              n_reps),
                  kpm::bench_moment_loop_fp32(H, p, kpm::GpuKernelMode::Naive,
                                              n_reps),
                  kpm::bench_moment_loop_fp32(H, p, kpm::GpuKernelMode::Fused,
                                              n_reps)};
      entries.push_back(e);

      const double bn64 = kpm::naive_bytes_per_moment(H.nnz(), H.n, 8);
      const double bf64 = kpm::fused_bytes_per_moment(H.nnz(), H.n, 8);
      const double tn = s_per_moment(e.fp64_naive, p.n_cheb, p.n_random);
      const double tf = s_per_moment(e.fp64_fused, p.n_cheb, p.n_random);
      const double gbps_n = bn64 / tn / 1e9, gbps_f = bf64 / tf / 1e9;
      std::printf("%4d %8d | %9.3f %9.3f | %9.3f %9.3f | %5.2fx | %7.1f "
                  "%7.1f | %5.1f%% %5.1f%%\n",
                  L, H.n, tn * 1e6, tf * 1e6,
                  s_per_moment(e.fp32_naive, p.n_cheb, p.n_random) * 1e6,
                  s_per_moment(e.fp32_fused, p.n_cheb, p.n_random) * 1e6,
                  tn / tf, gbps_n, gbps_f, 100.0 * gbps_n / triad.median_gbps,
                  100.0 * gbps_f / triad.median_gbps);
      std::printf("     spread us/moment [min,max]: n64 [%.3f,%.3f] f64 "
                  "[%.3f,%.3f] n32 [%.3f,%.3f] f32 [%.3f,%.3f]\n",
                  e.fp64_naive.min_s / (p.n_cheb * p.n_random) * 1e6,
                  e.fp64_naive.max_s / (p.n_cheb * p.n_random) * 1e6,
                  e.fp64_fused.min_s / (p.n_cheb * p.n_random) * 1e6,
                  e.fp64_fused.max_s / (p.n_cheb * p.n_random) * 1e6,
                  e.fp32_naive.min_s / (p.n_cheb * p.n_random) * 1e6,
                  e.fp32_naive.max_s / (p.n_cheb * p.n_random) * 1e6,
                  e.fp32_fused.min_s / (p.n_cheb * p.n_random) * 1e6,
                  e.fp32_fused.max_s / (p.n_cheb * p.n_random) * 1e6);
    }

    std::ofstream f(out_path);
    if (!f)
      throw std::runtime_error("cannot open " + out_path);
    f.precision(17);
    f << "{\n"
      << "  \"timestamp_utc\": \"" << timestamp_utc() << "\",\n"
      << "  \"seed\": \"" << hex_seed(p.seed) << "\",\n"
      << "  \"device\": \"" << device << "\",\n"
      << "  \"device_info\": {\n"
      << "    \"name\": \"" << devinfo.name << "\",\n"
      << "    \"compute_capability\": \"" << devinfo.major << "."
      << devinfo.minor << "\",\n"
      << "    \"sm_count\": " << devinfo.sm_count << ",\n"
      << "    \"clock_rate_khz\": " << devinfo.clock_rate_khz << ",\n"
      << "    \"memory_clock_rate_khz\": " << devinfo.memory_clock_rate_khz
      << ",\n"
      << "    \"memory_bus_width_bits\": " << devinfo.memory_bus_width_bits
      << ",\n"
      << "    \"l2_bytes\": " << devinfo.l2_bytes << ",\n"
      << "    \"global_mem_bytes\": " << devinfo.global_mem_bytes << ",\n"
      << "    \"driver_version\": " << devinfo.driver_version << ",\n"
      << "    \"runtime_version\": " << devinfo.runtime_version << "\n"
      << "  },\n"
      << "  \"toolchain\": \"" << kpm::gpu_toolchain() << "\",\n"
      << "  \"n_cheb\": " << p.n_cheb << ",\n"
      << "  \"n_random\": " << p.n_random << ",\n"
      << "  \"n_reps\": " << n_reps << ",\n"
      << "  \"triad_gbps\": {\"median\": " << triad.median_gbps
      << ", \"min\": " << triad.min_gbps << ", \"max\": " << triad.max_gbps
      << ", \"reps\": " << triad.n_reps << ",\n"
      << "    \"reps_gbps\": [";
    for (size_t i = 0; i < triad.reps_gbps.size(); ++i)
      f << (i ? ", " : "") << triad.reps_gbps[i];
    f << "]},\n"
      << "  \"sizes\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
      const SizeEntry& e = entries[i];
      f << "    {\"L\": " << e.L << ", \"n\": " << e.n
        << ", \"nnz\": " << e.nnz << ",\n"
        << "     \"fp64\": {\"naive\": ";
      json_result(f, e.fp64_naive, p.n_cheb, p.n_random,
                  kpm::naive_bytes_per_moment(e.nnz, e.n, 8), 16);
      f << ",\n     \"fused\": ";
      json_result(f, e.fp64_fused, p.n_cheb, p.n_random,
                  kpm::fused_bytes_per_moment(e.nnz, e.n, 8), 16);
      f << "},\n     \"fp32\": {\"naive\": ";
      json_result(f, e.fp32_naive, p.n_cheb, p.n_random,
                  kpm::naive_bytes_per_moment(e.nnz, e.n, 4), 16);
      f << ",\n     \"fused\": ";
      json_result(f, e.fp32_fused, p.n_cheb, p.n_random,
                  kpm::fused_bytes_per_moment(e.nnz, e.n, 4), 16);
      f << "}}" << (i + 1 < entries.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    if (!f)
      throw std::runtime_error("write failed: " + out_path);
    std::printf("\nJSON -> %s\n", out_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_kpm failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
