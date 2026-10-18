# kpm-tightbinding

This repository computes the density of states of a two-dimensional square-lattice
tight-binding model with the Kernel Polynomial Method (KPM). It includes an
exact-diagonalization reference, a stochastic CPU implementation, and hand-written
naive and fused GPU kernels. The GPU experiments focus on memory traffic, launch
overhead, and cache effects.

The measured path contains no vendor library kernels. Eigen is used only for the
host exact-diagonalization reference.

## Repository contents

- `docs/KPM.md`: lattice, KPM equations, and numerical tolerances
- `docs/R7_WRITEUP.md`: technical report and experimental results
- `src/`: lattice construction, exact diagonalization, CPU KPM, and GPU kernels
- `tests/`: numerical and benchmark-model tests
- `benchmarks/`: naive/fused timing driver and STREAM-like triad baseline
- `tools/`: DOS and scaling plots
- `data/`: CSV, JSON, and plot outputs
- `r5_hip/`: hipified source and build instructions
- `r6_amd/`: local Radeon runtime investigation

## Numerical validation

The CPU and GPU implementations are checked against exact diagonalization and
against each other. Failures block use of the solver output. Benchmark tables
report fused and naive results whether or not fusion wins.

Exact diagonalization for L=8..14 gives:

- max |E| = 3.758770483143635 at L=8, inside [-4, 4];
- |Tr H| <= 8.5e-14 from the summed eigenvalues;
- bipartite-symmetry errors of 8.0e-15 at L=8 and 2.6e-14 at L=14;
- E_min(L=14) = -3.912590402935264, compared with
  -4cos(pi/15) = -3.912590402935223.

The CPU KPM run uses L=14, 1000 Chebyshev moments, R=32 random vectors,
a=4.04, Jackson damping, and a fixed seed:

- DOS integral: 0.9999996, an error of 3.9e-7;
- mean DOS symmetry error: 0.0174;
- mu_2 = -0.57073, compared with the analytic -0.54486;
- matched-bin L1 distance from the exact-diagonalization histogram: 0.1953;
- integrated weight for |E| < 0.2: 0.0957, compared with 0.0918 for the
  histogram;
- moment-convergence errors versus R=128, reported as linf/l2:
  0.1176/1.0024 at R=4 and 0.0433/0.4317 at R=32.

The reference CSVs and DOS plot are in `data/`. The L1 tolerance is a regression
bound, not an accuracy estimate: with 196 eigenvalues, histogram sampling noise
is the dominant contribution.

## GPU correctness

For the same seed and R=32, FP64 GPU moments match the CPU result to
linf 1.1e-16. The random vectors and state vectors are bitwise identical by
construction; only the dot-reduction order differs. Both kernels are checked at
L=14; the fused kernel is also checked at L=8, where the lattice has fewer
sites than one 256-thread block. Multi-block grids are covered at L=17
(2 blocks, both kernels, linf 3.3e-16, with the two GPU modes differing by
0.0 on this run)
and at benchmark-class sizes L=128 (64 blocks) and L=256 (256 blocks) with
reduced moment counts, linf 1.3e-14 and 6.6e-14.

FP32 moments are within 1.9e-6 of the CPU FP64 reference. FP64 remains the
default for the physics runs. CTest passes 2/2, including the analytic-traffic
regression test.

## NVIDIA benchmark

The latest complete sweep used an RTX 5060 Laptop GPU (`sm_120`, 26 SMs,
32 MiB L2), 1000 moments, R=4, device-side timing, and 11 repetitions per point.
The per-moment `<r|t>` reduction is included in both modes. The measured FP64
STREAM-like triad baseline is 347.9 GB/s, median [346.9, 348.8] over 20
repetitions.

FP64 microseconds per Chebyshev moment, median [min,max]:

| L | N | naive | fused | naive/fused median-time ratio |
|--:|--:|--:|--:|--:|
| 16 | 256 | 7.28 [7.10, 7.74] | 4.82 [4.78, 4.84] | 1.51x |
| 32 | 1024 | 7.19 [7.10, 7.25] | 4.81 [4.80, 5.01] | 1.49x |
| 64 | 4096 | 7.24 [7.18, 7.26] | 4.82 [4.80, 4.85] | 1.50x |
| 128 | 16384 | 7.35 [7.34, 7.44] | 6.58 [6.58, 6.58] | 1.12x |
| 256 | 65536 | 17.39 [17.38, 17.40] | 16.21 [16.21, 17.06] | 1.07x |
| 512 | 262144 | 55.15 [55.14, 55.17] | 54.14 [54.05, 54.19] | 1.02x |

At small sizes, three naive kernel launches per moment versus two fused launches
put both paths on a latency floor. At larger sizes, the traffic model alone does
not predict the timing ratio: this sweep narrows toward 1.0 at L=512, while the
previous sweep showed an L=256 reversal at 0.84x. The large-size ratio is not
stable across runs; controlled repetitions are required before any general
large-size conclusion.

The FP64 analytic traffic per moment is `matrix + 56n` for the naive path and
`matrix + 40n` for the fused path. Fusion removes the global-memory round trip
of the intermediate `w`; it does not remove a second `t_n` read. The resulting
vector-traffic ratio is 1.40, or 1.17 when matrix traffic is included. These are
algorithmic byte counts, not hardware-counter measurements.

Naive FP64 effective bandwidth reaches 147% of the measured triad at L=512 and
117% at L=256, which is consistent with part of the traffic being served from
L2. At L=512, fused FP32 takes 35.44 [35.44, 35.45] microseconds versus
54.14 [54.05, 54.19] for FP64, a ratio of 1.53 rather than 2.

The full FP32/FP64 results, including per-repetition raw timings and
device/driver/seed/toolchain metadata, are in `data/kpm_gpu_bench.json`; the
scaling plot is `data/kpm_gpu_scaling.png`. Runs differ materially with machine
state — a cold-start sweep showed small-size moment times roughly 4x higher
with the triad unchanged — so the table should be read as one measured sweep
with observed spreads, not a stable speedup distribution.

## HIP and local AMD results

`hipify-perl` mapped the CUDA calls in the kernel source 1:1. The hipified copy
compiles with `hipcc 7.1.1` for `gfx942`; the only manual edit was a local include
path. This establishes source compilation, not runtime correctness or
performance on MI300 hardware.

The local runtime investigation used an integrated Radeon 610M (`gfx1036`). A
minimal store probe matched the upstream GC 10.3.6 GFX power-gating/dirty-L2
failure. With `AMD_OPT_FLUSH=0`, the probe and full current suite passed 5/5,
followed by an audited six-size benchmark. The triad baseline was 29.97 GB/s and
the fused ratios ranged from 1.45x to 1.12x. These results are specific to that
runtime environment. MI300X and MUSA hardware remain untested.

See `r5_hip/README.md` and `r6_amd/README.md` for commands and evidence.

## Build and test

The recorded NVIDIA toolchain is CUDA 13.2 with an RTX 5060 Laptop GPU. The
CMake configuration selects g++-15 for host compilation because nvcc rejects the
installed g++-16 headers.

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build
ctest --test-dir build --output-on-failure
```
