#!/usr/bin/env python3
"""Overlay KPM DOS vs exact-diagonalization histogram (L=14), plus the
infinite-lattice analytic DOS rho(E) = K(1 - (E/4)^2) / (2 pi^2) with K
the complete elliptic integral of the first kind (modulus convention,
via AGM). Reads data/, writes data/dos_L14.png."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", skip_header=1)


def elliptic_k_modulus(k):
    """K(k) = integral_0^{pi/2} dtheta / sqrt(1 - k^2 sin^2 theta) (AGM)."""
    a = np.ones_like(k)
    g = np.sqrt(1.0 - k**2)
    for _ in range(30):
        a, g = 0.5 * (a + g), np.sqrt(a * g)
    return np.pi / (2.0 * a)


def analytic_dos(E):
    k = 1.0 - (E / 4.0) ** 2
    return elliptic_k_modulus(k) / (2.0 * np.pi**2)


def main():
    kpm = load_csv(DATA / "kpm_dos_L14.csv")
    hist = load_csv(DATA / "exact_dos_L14.csv")

    Ea = np.linspace(-3.999, 3.999, 2000)
    rho_a = analytic_dos(Ea)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kpm[:, 0], kpm[:, 1], lw=0.8, alpha=0.6,
            label="KPM (N_cheb=1000, R=32, Jackson)")
    # The N_cheb=1000 resolution (~0.013 in E) is finer than the L=14 level
    # spacing, so the raw KPM curve resolves individual eigenvalue spikes.
    # A rolling mean (~0.1 window) shows the smooth envelope.
    w = max(1, int(round(0.1 / np.median(np.diff(kpm[:, 0])))))
    sm = np.convolve(kpm[:, 1], np.ones(w) / w, mode="same")
    ax.plot(kpm[:, 0], sm, lw=1.6, color="k", label="KPM, rolling mean (~0.1)")
    dE = hist[1, 0] - hist[0, 0]
    ax.bar(hist[:, 0], hist[:, 1], width=dE, alpha=0.5,
           label="exact diag histogram (L=14)")
    ax.plot(Ea, rho_a, lw=1.0, ls="--", label="analytic (infinite lattice)")
    ax.set_xlabel("E")
    ax.set_ylabel("rho(E)")
    ax.set_title("2D square-lattice tight-binding DOS, L=14, open BCs")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    out = DATA / "dos_L14.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
