#include "kpm/kpm.hpp"
#include "kpm/lattice.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  int L = 14;
  kpm::KpmParams p;
  int n_grid = 2048;
  for (int i = 1; i < argc; ++i) {
    try {
      if (std::strncmp(argv[i], "--L=", 4) == 0)
        L = std::stoi(argv[i] + 4);
      else if (std::strncmp(argv[i], "--n-cheb=", 9) == 0)
        p.n_cheb = std::stoi(argv[i] + 9);
      else if (std::strncmp(argv[i], "--R=", 4) == 0)
        p.n_random = std::stoi(argv[i] + 4);
      else if (std::strncmp(argv[i], "--seed=", 7) == 0)
        p.seed = std::stoull(argv[i] + 7);
      else if (std::strncmp(argv[i], "--grid=", 7) == 0)
        n_grid = std::stoi(argv[i] + 7);
      else
        throw std::invalid_argument(std::string("unknown argument: ") +
                                    argv[i]);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dos_kpm: %s\n", e.what());
      std::fprintf(stderr, "usage: dos_kpm [--L=14] [--n-cheb=1000] "
                           "[--R=32] [--seed=N] [--grid=2048]\n");
      return 1;
    }
  }

  try {
    const kpm::CsrMatrix H = kpm::square_lattice(L);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<double> mu = kpm::chebyshev_moments(H, p);
    const auto t1 = std::chrono::steady_clock::now();
    const std::vector<double> g = kpm::jackson_kernel(p.n_cheb);
    const auto dos = kpm::kpm_dos(mu, g, p.a, p.b, n_grid);
    const auto t2 = std::chrono::steady_clock::now();

    const std::string path =
        std::string(KPM_DATA_DIR) + "/kpm_dos_L" + std::to_string(L) + ".csv";
    std::ofstream f(path);
    if (!f)
      throw std::runtime_error("cannot open " + path);
    f << "E,rho\n";
    f.precision(17);
    for (const auto& [E, rho] : dos)
      f << E << ',' << rho << '\n';
    if (!f)
      throw std::runtime_error("write failed: " + path);

    std::printf("KPM L=%d N=%d n_cheb=%d R=%d a=%.4f b=%.1f seed=%llu\n", L,
                H.n, p.n_cheb, p.n_random, p.a, p.b,
                (unsigned long long)p.seed);
    std::printf("mu_0..3 = %.12f %+.3e %+.12f %+.3e\n", mu[0], mu[1], mu[2],
                mu[3]);
    std::printf("integral rho dE = %.9f (target 1)\n", kpm::integrate_dos(dos));
    std::printf("moments %.1f ms, dos grid %.1f ms -> %s\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count(),
                path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dos_kpm failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
