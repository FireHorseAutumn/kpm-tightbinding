# CUDA-to-HIP compile check

This directory contains the hipify-perl output for `src/kpm_gpu.cu` and
`include/kpm/kpm_gpu.hpp`, plus the shared naive-traffic and min/max reporting
corrections made after the original port. Those corrections are not HIP porting
changes.

Regenerate the original port from the repository root with:

```bash
cp src/kpm_gpu.cu r5_hip/ && cp include/kpm/kpm_gpu.hpp r5_hip/
hipify-perl -i r5_hip/kpm_gpu.cu && hipify-perl -i r5_hip/kpm_gpu.hpp
# Adjust the local include path in kpm_gpu.cu, then:
hipcc -std=c++17 -Iinclude -Ir5_hip --offload-arch=gfx942 -c r5_hip/kpm_gpu.cu
```

## Observed build friction

1. hipify-perl mapped every CUDA API call in the kernel file 1:1. This includes
   allocation, copies, events, error handling, device properties, `atomicAdd`,
   and `__syncthreads`. No manual kernel edits were needed for the original port.
2. The hipified copy needed its local include changed from `kpm/kpm_gpu.hpp` to
   `kpm_gpu.hpp` because of the directory layout.
3. The source compiled with hipcc 7.1.1 for `gfx942`. The remaining warnings
   concern unchecked `hipFree` and `hipEventDestroy` return values in cleanup
   paths.
4. Fedora's `rocm-hip-devel` package does not include the hipify tools; the
   `hipify` package supplies hipify-clang and hipify-perl.

The kernels use FP64 and mainstream CUDA runtime APIs, with no cooperative
groups, dynamic parallelism, texture objects, cuBLAS, or cuSPARSE. The compile
result applies to this source profile only.

Runtime testing later exposed store-visibility and link-configuration issues on
a local Radeon 610M. See `../r6_amd/README.md`; MI300X remains untested.
