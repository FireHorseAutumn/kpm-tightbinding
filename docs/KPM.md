# KPM mathematical specification

This document defines the lattice, Chebyshev expansion, GPU traffic model, and
numerical checks used in the repository. The main physics reference is Weiße,
Wellein, Alvermann, and Fehske, Rev. Mod. Phys. 78, 275 (2006).

## Lattice and exact reference

The model is an L×L square lattice with N=L² sites, nearest-neighbor hopping
t=-1, zero onsite energy, and open boundaries. Energies are measured in |t|.

For open boundaries, the analytic eigenvalues are

    E = -2(cos k_x + cos k_y),
    k_i = n_i pi/(L+1),  n_i = 1..L.

The spectrum lies in (-4,4), is symmetric about zero, and satisfies Tr H=0 and
E_min(L)=-4cos(pi/(L+1)). The CSR builder uses row-major site index
`(i,j) -> i*L+j` and at most five entries per row.

Dense eigensolves for L=8..14 use Eigen's `SelfAdjointEigenSolver`. The checks
are |E| <= 4+epsilon, trace and bipartite symmetry near machine precision, and
E_min(L=14)=-4cos(pi/15)=-3.91259... to 1e-12. Eigen is not used in the measured
KPM path.

## Chebyshev expansion

1. Rescale `H'=(H-bI)/a` so that `spec(H')` is contained in [-1,1]. The default
   is a=4(1+eta), b=0, and eta=0.01.
2. Estimate moments with R random +/-1 vectors:

       mu_n = (1/N)(1/R) sum_r <r|T_n(H')|r>
       t_0 = r
       t_1 = H'r
       t_(n+1) = 2H't_n - t_(n-1)

   The reference run uses R=32; convergence is compared against R=4 and R=128.
3. Apply Jackson damping for n=0..N_cheb, following Eq. 71 of the reference:

       g_n = [(N_cheb-n+1) cos(pi n/(N_cheb+1))
             + sin(pi n/(N_cheb+1)) cot(pi/(N_cheb+1))]/(N_cheb+1)

4. Reconstruct the DOS for epsilon in (-1,1):

       rho'(epsilon) = [g_0 mu_0 + 2 sum_(n=1)^(N_cheb-1) g_n mu_n T_n(epsilon)]
                       / [pi sqrt(1-epsilon^2)]
       rho(E) = rho'(E/a)/a

   Evaluate `T_n` by recurrence rather than `cos(n arccos epsilon)`. The default
   expansion order is N_cheb=1000.

## Numerical checks

- The trapezoidal DOS integral agrees with one to approximately 1e-3.
- `rho(E)` agrees with `rho(-E)` up to stochastic noise.
- The L1 distance from the L=14 exact-diagonalization histogram uses matched
  bins. Because the exact DOS diverges logarithmically at E=0 while Jackson
  damping gives a finite KPM peak, the comparison near zero uses integrated
  weight in a window rather than the point value.
- An optional analytic reference is the infinite-lattice DOS
  `rho(E)=(1/(2 pi^2)) K(1-(E/4)^2)`, where K is the complete elliptic integral
  of the first kind. Compare away from E=0 and +/-4.

## Naive GPU path

Each moment uses a CSR SpMV kernel, `w=H't_n`, followed by a recurrence kernel,
`t_(n+1)=2w-t_(n-1)`. The dot-product kernel then reads `r` and `t_(n+1)`.

With `rb=sizeof(Real)` and `n=L²`, the analytic global-memory total per moment
and random vector is `matrix + 7 rb n`, including the dot product. The benchmark
sweep covers L=16..512 to include launch-latency and larger-cache regimes.

## Fused GPU path

The fused kernel computes each row's SpMV result and applies the recurrence in
registers, so `w` never reaches global memory. It reads `t_n`, `t_(n-1)`, and
the matrix, then writes `t_(n+1)`. Including the separate dot product, the
analytic total is `matrix + 5 rb n`.

Benchmarks compare this path with the naive path using medians, observed spreads,
effective GB/s, and a measured STREAM-like baseline on the same GPU. FP32 and
FP64 are reported separately.

## HIP

The hipified kernels are compile-checked with:

```bash
hipcc --offload-arch=gfx942
```

The compile check does not establish runtime behavior. `r6_amd/` documents the
local Radeon 610M investigation and the results obtained with
`AMD_OPT_FLUSH=0`. Those results should not be transferred to other runtime
environments or AMD devices without new measurements. MI300X remains untested.

## Reproducibility

Check every kernel launch and stop on verification failure or non-convergence.
Report medians and observed spreads together with the machine state and runtime
conditions. Negative and near-tie results belong in the benchmark tables alongside
improvements.
