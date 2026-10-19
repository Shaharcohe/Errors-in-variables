"""Heatmaps of the two design covariances Sigma_x (AR and CS), side by side.

    python plot_covariance_heatmaps.py              # -> results/covariance_heatmaps.{png,pdf}
    python plot_covariance_heatmaps.py --p 30

Both matrices have the same pattern at every p (AR is Toeplitz, CS is constant
off the diagonal), so the leading p x p block of the p=500 matrix is shown.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from config import sigma_x_ar, sigma_x_cs

INK = "#333333"
# one-hue sequential ramp, near-white at 0 -> dark blue at 1
CMAP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
PANELS = [
    (sigma_x_ar, r"AR:  $\Sigma_{ij} = 0.5^{|i-j|}$"),
    (sigma_x_cs, r"CS:  $\Sigma_{ij} = 0.5 + 0.5\,\mathbb{1}\{i=j\}$"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=20, help="size of the block shown")
    ap.add_argument("--out", default="results/covariance_heatmaps")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    ticks = [0] + list(range(4, args.p, 5))
    for ax, (sigma_x, title) in zip(axes, PANELS):
        S = sigma_x(args.p)
        mesh = ax.pcolormesh(S, cmap=CMAP, vmin=0, vmax=1,
                             edgecolors="white", linewidth=0.6)
        ax.set_xlim(0, args.p)
        ax.set_ylim(args.p, 0)  # row 1 at the top, like a matrix
        ax.set_aspect("equal")
        ax.set_xticks(np.array(ticks) + 0.5)
        ax.set_xticklabels([t + 1 for t in ticks])
        ax.set_yticks(np.array(ticks) + 0.5)
        ax.set_yticklabels([t + 1 for t in ticks])
        ax.tick_params(length=0, labelsize=9, colors=INK)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
        ax.set_xlabel("$j$", color=INK)
        ax.set_ylabel("$i$", color=INK)

    cbar = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=9, colors=INK)
    cbar.set_label(r"$\Sigma_{ij}$", color=INK)
    fig.suptitle(f"Design covariance $\\Sigma_x$ (leading {args.p}$\\times${args.p} block)",
                 fontsize=13, fontweight="bold", color=INK, x=0.06, ha="left")

    for ext in ("png", "pdf"):
        path = f"{args.out}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
