# Local AMD HIP runtime investigation

Updated 2026-10-17.

The tests in this directory used an integrated AMD Radeon 610M (`gfx1036`) with
HIP/hipcc 7.1.1. On this machine, `AMD_OPT_FLUSH=0` is required to reproduce the
passing probe, KPM correctness runs, and benchmark. The results do not describe
MI300X or other AMD devices.

## Host and toolchain

The host exposes `/dev/kfd` and one AMD device. ROCm SMI identifies Raphael
`gfx1036`; HIP reports `AMD Radeon 610M`. The hipcc link requires explicit flags:

```text
-L/usr/lib64 -lamdhip64 -lhsa-runtime64
```

The installation has no `rocm_agent_enumerator`, so the architecture is specified
explicitly. `rocminfo` is absent. `rocm-smi` prints a libdrm warning although
`ldd` resolves the versioned library; neither observation was established as the
cause of the numerical failures.

## Failure and diagnosis

The hipified source compiled and launched, but the KPM numerical checks failed
intermittently. Small cases could pass while larger R=32 runs lost or corrupted
contributions. `HIP_LAUNCH_BLOCKING=1` did not make the results reliable.

The R6 source adds synchronization after initial host-to-device uploads, after
moment-accumulator initialization, around device-to-device random-state
initialization, and before final result readback. Copy-kernel and non-atomic
variants did not fix the failures and were removed.

Earlier five-pass correctness results and a benchmark were followed by failures
in fresh runs of the same suite. A performance-mode test passed once and then
failed with the same probe signature. `AMD_SERIALIZE_KERNEL=3` passed the minimal
probe 5/5 but failed the first full KPM run. The earlier 31.95 GB/s triad and
1.15-1.30x fusion ratios are therefore retained only as historical observations.

`mu0_store_probe.cpp` reduces the problem to a serial dot product whose expected
result is 196. Separate guards record kernel entry and completion after the
store. Outputs and guards start as NaN, and every HIP call is checked. Without
the workaround, synchronized runs could return successful HIP statuses while the
result and both guards remained NaN. The no-inner-sync control passed; with
`AMD_OPT_FLUSH=0`, the synchronized probe passed 5/5.

The signature matches [ROCm Systems issue #13469](https://github.com/ROCm/rocm-systems/issues/13469)
and the linked [drm/amd report](https://gitlab.freedesktop.org/drm/amd/-/issues/5923).
Those reports describe GC 10.3.6 behavior in which an agent-scope release fence
followed by coarse-grain GFX power gating can lose dirty L2 contents. The upstream
discussion identifies system-scope releases and GFX power-gating changes as
mitigations. No kernel or firmware setting was changed for these experiments.

## Minimal probe

From the repository root:

```bash
hipcc --offload-arch=gfx1036 -std=c++20 -Wno-unused-result \
  r6_amd/mu0_store_probe.cpp -o build/mu0_store_probe \
  -L/usr/lib64 -lamdhip64 -lhsa-runtime64
hipcc --offload-arch=gfx1036 -std=c++20 -Wno-unused-result \
  -DNO_INNER_SYNC r6_amd/mu0_store_probe.cpp \
  -o build/mu0_store_probe_nosync -L/usr/lib64 -lamdhip64 -lhsa-runtime64
AMD_OPT_FLUSH=0 LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64 \
  ./build/mu0_store_probe 455
LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64 \
  ./build/mu0_store_probe_nosync 455
```

The first command runs the synchronized probe with the workaround. The second is
the diagnostic control.

## KPM correctness

```bash
hipcc --offload-arch=gfx1036 -std=c++20 -Wno-unused-result \
  -Iinclude -Ir6_amd \
  src/lattice.cpp src/kpm.cpp r6_amd/kpm_gpu.cu tests/test_gpu_kpm.cpp \
  -o build/test_gpu_kpm_amd -L/usr/lib64 -lamdhip64 -lhsa-runtime64
AMD_OPT_FLUSH=0 LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64 \
  ./build/test_gpu_kpm_amd
```

Five independent processes passed, including the L=8 and L=14 paths. The
reported errors were 1.110e-16 for FP64 and 1.859e-06 for FP32. Keep
`AMD_OPT_FLUSH=0` set when reproducing these local correctness runs.

## Benchmark

```bash
hipcc --offload-arch=gfx1036 -std=c++20 -Wno-unused-result \
  -DKPM_DATA_DIR=\"$(pwd)/data\" -Iinclude -Ir6_amd \
  src/lattice.cpp src/kpm.cpp r6_amd/kpm_gpu.cu benchmarks/bench_kpm.cpp \
  -o build/bench_kpm_amd -L/usr/lib64 -lamdhip64 -lhsa-runtime64
AMD_OPT_FLUSH=0 LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64 \
  ./build/bench_kpm_amd --sizes=16,32,64,128,256,512 \
  --n-cheb=1000 --R=4 --reps=11 --triad-reps=20 \
  --out=data/kpm_gpu_bench_amd_gfx1036.json
```

The previous JSON was saved as
`build/kpm_gpu_bench_amd_gfx1036_pre_opt_flush.json`. The new JSON passed checks
of every `min <= median <= max` invariant and all byte-count, per-moment, and
effective-GB/s formulas. It contains aggregated summaries rather than raw
repetitions. Its CUDA-centric device string (`sm_103, 1 SMs, L2 0 MiB`) should
not be interpreted literally for `gfx1036`.

The FP64 STREAM-like triad result is 29.971365 GB/s [29.153302, 30.589474],
median [min,max] over 20 repetitions.

FP64 microseconds per Chebyshev moment, median [min,max]:

| L | naive | fused | naive/fused median-time ratio |
|--:|--:|--:|--:|
| 16 | 21.171 [19.010, 22.392] | 14.553 [13.900, 14.970] | 1.4547x |
| 32 | 19.710 [19.188, 20.329] | 13.764 [13.235, 14.155] | 1.4320x |
| 64 | 31.285 [31.001, 31.454] | 26.110 [25.935, 26.402] | 1.1982x |
| 128 | 99.376 [97.483, 107.329] | 86.824 [83.717, 89.350] | 1.1446x |
| 256 | 357.036 [352.016, 384.106] | 306.611 [303.806, 380.687] | 1.1645x |
| 512 | 1582.137 [1507.286, 1648.786] | 1413.229 [1360.651, 1608.412] | 1.1195x |

The spreads are substantial, and the two GPUs discussed in this repository are
not equivalent hardware.

## Logs

The local evidence logs are:

- `build/r6_opt_flush_probe.log`
- `build/r6_opt_flush_kpm.log`
- `build/r6_opt_flush_benchmark.log`
- `build/r6_opt_flush_benchmark_audit.log`
- `build/r6_performance_probe.log`

If MI300X access becomes available, run correctness tests before benchmarking.
The Radeon diagnosis and synchronization results need independent verification on
that hardware.
