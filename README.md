# kpm-tightbinding

Density of states of a two-dimensional square-lattice tight-binding model,
computed with the Kernel Polynomial Method (KPM): exact-diagonalization
reference, stochastic CPU implementation, and hand-written naive and fused
GPU kernels, with experiments focused on memory traffic, launch overhead,
and cache effects. No vendor library kernels on the measured path; Eigen is
used only for the host exact-diagonalization reference.

## Status

- CTest 2/2, including the analytic-traffic regression test; FP64 GPU
  moments match CPU to linf 1.1e-16 (same seed, R=32).
- Recorded with CUDA 13.2 on an RTX 5060 Laptop GPU (`sm_120`, 26 SMs,
  32 MiB L2): 1000 moments, R=4, device-side timing, 11 repetitions per
  point. Triad baseline: 347.9 GB/s, median [346.9, 348.8] over 20 runs.

## Headline: naive vs fused microseconds per moment

FP64 microseconds per Chebyshev moment, median [min,max]:

| L | N | naive | fused | naive/fused median-time ratio |
|--:|--:|--:|--:|--:|
| 16 | 256 | 7.28 [7.10, 7.74] | 4.82 [4.78, 4.84] | 1.51x |
| 32 | 1024 | 7.19 [7.10, 7.25] | 4.81 [4.80, 5.01] | 1.49x |
| 64 | 4096 | 7.24 [7.18, 7.26] | 4.82 [4.80, 4.85] | 1.50x |
| 128 | 16384 | 7.35 [7.34, 7.44] | 6.58 [6.58, 6.58] | 1.12x |
| 256 | 65536 | 17.39 [17.38, 17.40] | 16.21 [16.21, 17.06] | 1.07x |
| 512 | 262144 | 55.15 [55.14, 55.17] | 54.14 [54.05, 54.19] | 1.02x |

The fused-dot mode folds the `<r|t>` reduction into the fused kernel: 1.28x
over fused at L=16 to L=64, decaying to 1.02x at L=512.

## Key numbers

- Exact diagonalization, L=8..14: max |E| = 3.758770483143635 at L=8, inside
  [-4, 4]; |Tr H| <= 8.5e-14; E_min(L=14) = -3.912590402935264 versus
  -4cos(pi/15) = -3.912590402935223.
- CPU KPM (L=14, 1000 moments, R=32, a=4.04, Jackson damping, fixed seed):
  DOS integral 0.9999996, an error of 3.9e-7.
- FP64 analytic traffic per moment: `matrix + 56n` naive, `matrix + 40n`
  fused; ratio 1.40, or 1.17 with matrix traffic. Nsight counters at L=512
  match: fused saves 4.21 MB LTS, exactly one w write + reread (2 x 2.097
  MB); the dot costs 4.20 MB DRAM, exactly two vector reads.
- Naive FP64 effective bandwidth reaches 147% of the measured triad at L=512
  and 117% at L=256, consistent with part of the traffic served from L2.

## Honest limits

- Large-size ratio not stable across runs: this sweep narrows toward 1.0 at
  L=512; the previous sweep showed an L=256 reversal at 0.84x. One measured
  sweep with observed spreads, not a stable distribution: a cold-start sweep
  showed small-size times roughly 4x higher with the triad unchanged.
- The hipified copy compiles with `hipcc 7.1.1` for `gfx942`: source
  compilation only, not runtime correctness or performance on MI300
  hardware. The Radeon 610M (`gfx1036`) results are specific to that
  runtime; MI300X and MUSA hardware remain untested.
- The L1 tolerance is a regression bound, not an accuracy estimate: with 196
  eigenvalues, histogram sampling noise is the dominant contribution.

## Build and test

The CMake configuration selects g++-15 for host compilation because nvcc
rejects the installed g++-16 headers.

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build
ctest --test-dir build --output-on-failure
```

## Repository layout

- `docs/KPM.md` and `docs/R7_WRITEUP.md`: model, equations, tolerances; technical report
- `src/`: lattice construction, exact diagonalization, CPU KPM, and GPU kernels
- `tests/`: numerical and benchmark-model tests
- `benchmarks/`: naive/fused timing driver and STREAM-like triad baseline
- `tools/`: DOS and scaling plots
- `data/`: CSV, JSON, and plot outputs
- `r5_hip/` and `r6_amd/`: hipified source; local Radeon runtime investigation

## Deeper reading

- `docs/RUNGS.md`: the full record: numerical validation, GPU correctness
  across grid sizes, the benchmark narrative, and the HIP/AMD results.
- `data/kpm_gpu_bench.json`: FP32/FP64 raw timings and metadata; scaling
  plot `data/kpm_gpu_scaling.png`. HIP evidence: `r5_hip/README.md` and
  `r6_amd/README.md`.
