"""Error histograms per sparsity setting of the p=500 sweep.

    python plot_error_hist.py                       # every sparsity, MSE and PE
    python plot_error_hist.py --metric mse --sparsity datta_zhang eta2.0

One figure per sparsity (datta_zhang, weak eta=1.0/1.5/2.0); one panel per
(model, noise) at low and high noise (sigma_a = 0.75, 1.25).
Each panel overlays CoCoLasso-max, ours-frobenius and ours-max over the
repetitions all three share (normally 100), on shared bins; dashed lines mark
the medians.
Uses only csv + numpy (pandas is not installed on the login node).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_IN = "results/sweep_p500_runs.csv"

ARMS = [("cocolasso_max", "CoCoLasso max", "#2a78d6"),
        ("reweighted_frobenius", "ours frob", "#eb6834"),
        ("reweighted_max", "ours max", "#1baf7a")]
MODELS = [("ar", "AR"), ("cs", "CS")]
NOISES = [("0.75", "low noise (σₐ=0.75)"), ("1.25", "high noise (σₐ=1.25)")]
# tag -> (beta_preset, alpha, title); alpha is the weak-sparsity decay eta.
SPARSITY = {
    "datta_zhang": ("datta_zhang", "1.0", "Datta & Zhang β* (exactly 3-sparse)"),
    "eta1.0": ("weak_sparse", "1.0", "weak-sparse β*, η=1.0 (83% in first 3)"),
    "eta1.5": ("weak_sparse", "1.5", "weak-sparse β*, η=1.5 (97% in first 3)"),
    "eta2.0": ("weak_sparse", "2.0", "weak-sparse β*, η=2.0 (99% in first 3)"),
}
METRIC_LABEL = {"mse": "MSE  ‖β̂ − β*‖²", "pe": "PE (prediction error)"}
INK = "#333333"


def load(path):
    """(beta_preset, alpha, model, sigma_a, arm) -> {seed: row}."""
    arms = {a for a, _, _ in ARMS}
    out = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["arm"] in arms:
                key = (r["beta_preset"], r["alpha"], r["model"], r["sigma_a"],
                       r["arm"])
                out[key][int(r["seed"])] = r
    return out


def plot(data, sparsity, metric, out_path, bins=25):
    preset, alpha, title = SPARSITY[sparsity]
    fig, axes = plt.subplots(len(MODELS), len(NOISES), figsize=(11, 7.5),
                             dpi=150, sharey=True)
    for i, (model, mlab) in enumerate(MODELS):
        for j, (sig, nlab) in enumerate(NOISES):
            ax = axes[i, j]
            runs = [data[(preset, alpha, model, sig, a)] for a, _, _ in ARMS]
            seeds = sorted(set.intersection(*(set(r) for r in runs)))
            vals = [np.array([float(r[s][metric]) for s in seeds]) for r in runs]
            edges = np.linspace(min(v.min() for v in vals),
                                max(v.max() for v in vals), bins + 1)
            for v, (_, lab, col) in zip(vals, ARMS):
                ax.hist(v, bins=edges, histtype="stepfilled", color=col,
                        alpha=0.12, zorder=2)
                ax.hist(v, bins=edges, histtype="step", color=col,
                        linewidth=2, zorder=3)
                ax.axvline(np.median(v), color=col, linewidth=1.5,
                           linestyle=(0, (4, 2)), zorder=4)
            text = "\n".join(f"{lab}: median {np.median(v):.2f}"
                             for v, (_, lab, _) in zip(vals, ARMS))
            ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right",
                    va="top", fontsize=7.5, color=INK, linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="none", alpha=0.8))
            ax.set_title(f"{mlab}, {nlab}  (n={len(seeds)})", fontsize=10,
                         loc="left", color=INK)
            ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == len(MODELS) - 1:
                ax.set_xlabel(METRIC_LABEL[metric], fontsize=9.5)
            if j == 0:
                ax.set_ylabel("repetitions", fontsize=9.5)

    handles = [plt.Line2D([], [], color=c, linewidth=2, label=l) for _, l, c in ARMS]
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.5,
                              linestyle=(0, (4, 2)), label="median"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.005, 0.965))
    fig.suptitle(f"{metric.upper()} distribution, {title}  (p=500, n=100)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_IN)
    ap.add_argument("--metric", nargs="+", default=["mse", "pe"],
                    choices=["mse", "pe"])
    ap.add_argument("--sparsity", nargs="+", default=list(SPARSITY),
                    choices=list(SPARSITY))
    ap.add_argument("--bins", type=int, default=25)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args(argv)

    data = load(args.csv)
    for sp in args.sparsity:
        for m in args.metric:
            plot(data, sp, m, Path(args.outdir) / f"error_hist_{sp}_{m}.png",
                 args.bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
