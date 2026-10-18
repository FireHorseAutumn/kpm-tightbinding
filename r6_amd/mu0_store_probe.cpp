#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int N = 196;
constexpr int R = 32;
constexpr int MU_LEN = 1001;
constexpr int BLOCK = 256;

#define CHECK(call)                                                             \
  do {                                                                          \
    const hipError_t err = (call);                                              \
    if (err != hipSuccess) {                                                     \
      std::fprintf(stderr, "HIP error at line %d in %s: %s\n", __LINE__,       \
                   #call, hipGetErrorString(err));                               \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)

__global__ void dot(const double* u, const double* v, double* out,
                    double* entry, double* done) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *entry = 1.0;
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
      sum += u[i] * v[i];
    *out = sum;
    *done = 2.0;
  }
}

int main(int argc, char** argv) {
  const int trials = argc > 1 ? std::atoi(argv[1]) : 5000;
  std::vector<double> slab(N * R);
  std::mt19937_64 rng(12345);
  for (double& x : slab)
    x = (rng() & 1ULL) ? 1.0 : -1.0;

  double* d_slab = nullptr;
  double* d_mu = nullptr;
  CHECK(hipMalloc(&d_slab, sizeof(double) * slab.size()));
  CHECK(hipMalloc(&d_mu, sizeof(double) * MU_LEN));
  int failures = 0;
  for (int trial = 0; trial < trials; ++trial) {
    CHECK(hipMemcpy(d_slab, slab.data(), sizeof(double) * slab.size(),
                    hipMemcpyHostToDevice));
    CHECK(hipMemset(d_mu, 0xFF, sizeof(double) * MU_LEN));
    CHECK(hipDeviceSynchronize());
    for (int r = 0; r < R; ++r) {
#ifndef NO_INNER_SYNC
      CHECK(hipDeviceSynchronize());
#endif
      dot<<<1, BLOCK>>>(d_slab + r * N, d_slab + r * N, d_mu + r,
                        d_mu + 64 + r, d_mu + 128 + r);
      CHECK(hipGetLastError());
    }
    CHECK(hipDeviceSynchronize());
    double results[160];
    CHECK(hipMemcpy(results, d_mu, sizeof(results), hipMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < R; ++r)
      if (results[r] != N)
        ++bad;
    if (bad != 0) {
      ++failures;
      std::printf("trial=%d bad=%d ", trial, bad);
      for (int r = 0; r < R; ++r)
        if (results[r] != N)
          std::printf("r%d=%.17g entry=%.17g done=%.17g ", r, results[r],
                      results[64 + r], results[128 + r]);
      std::printf("\n");
    }
  }
  CHECK(hipFree(d_slab));
  CHECK(hipFree(d_mu));
  std::printf("store probe failures=%d/%d\n", failures, trials);
  return 0;
}
