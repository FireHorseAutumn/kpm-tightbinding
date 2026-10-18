#!/usr/bin/env python3
"""R3 vs R4 scaling plot: naive vs fused moment time and effective
bandwidth as % of the measured STREAM-like triad peak, from
data/kpm_gpu_bench.json. PNG out (default data/kpm_gpu_scaling.png)."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/kpm_gpu_bench.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/kpm_gpu_scaling.png"
    with open(src) as f:
        d = json.load(f)

    triad = d["triad_gbps"]["median"]
    L = [e["L"] for e in d["sizes"]]

    def series(prec, mode, key):
        return [e[prec][mode][key] for e in d["sizes"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax1.plot(L, series("fp64", "naive", "us_per_moment"), "o-", label="naive FP64")
    ax1.plot(L, series("fp64", "fused", "us_per_moment"), "s-", label="fused FP64")
    ax1.plot(L, series("fp32", "naive", "us_per_moment"), "o--", label="naive FP32")
    ax1.plot(L, series("fp32", "fused", "us_per_moment"), "s--", label="fused FP32")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(L, [str(x) for x in L])
    ax1.set_xlabel("lattice size L (N = L$^2$ sites)")
    ax1.set_ylabel("µs per Chebyshev moment (median)")
    ax1.set_title(f"R3 naive vs R4 fused — {d['n_cheb']} moments, R={d['n_random']}")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    for prec, mode, style in [
        ("fp64", "naive", "o-"),
        ("fp64", "fused", "s-"),
        ("fp32", "fused", "s--"),
    ]:
        gbps = series(prec, mode, "effective_gbps")
        ax2.plot(L, [100.0 * g / triad for g in gbps], style,
                 label=f"{mode} {prec.upper()}")
    ax2.axhline(100.0, color="k", lw=1, ls=":",
                label=f"triad peak ({triad:.0f} GB/s)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(L, [str(x) for x in L])
    ax2.set_xlabel("lattice size L (N = L$^2$ sites)")
    ax2.set_ylabel("effective GB/s, % of measured triad")
    ax2.set_title("Roofline position (>100% = served from L2, not DRAM)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(d["device"], fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
