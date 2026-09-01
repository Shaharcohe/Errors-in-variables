"""Per-setting plots from the per-repetition CSV written by export_runs_csv.py.

    python plot_settings.py                      # both plot kinds, every setting
    python plot_settings.py --metric mse
    python plot_settings.py --setting ar_sigma1.0_datta_zhang

Three figures per setting, written into that setting's own results directory:

  compare_all.png                every algorithm side by side, one point per
                                 repetition
  paired_coco_vs_rew.png         CoCoLasso against the reweighted algorithm at
                                 the same norm -- is the reweighting worth it?
  paired_rew_cheap_vs_coco_max.png
                                 the reweighted algorithm on a cheap norm
                                 against CoCoLasso on the max norm -- can the
                                 expensive projection be skipped entirely?

Both paired figures exploit the same fact. Every arm is run on the same seeds,
and seeding is deterministic given (seed, model, n, p, sigma_a, sigma_e,
beta*), so for a given seed two arms saw bit-identical data. Plotting one point
per seed keeps that pairing instead of throwing it away in two separate means,
and the y=x diagonal reads directly: below it, the arm on the y axis won that
dataset.

They differ in what is held fixed. paired_coco_vs_rew holds the norm and the
variant fixed (plain with plain, refit with refit) so the reweighting is the
only difference. paired_rew_cheap_vs_coco_max deliberately crosses norms: the
published baseline at its expensive max-norm projection against the reweighted
algorithm at a cheap one, which is the comparison that decides whether the ADMM
is needed at all.

Colour encodes the projection norm, never the algorithm: the categorical
palette only validates all-pairs colourblind separation for three slots, and
there are exactly three norms. The algorithm is carried by position instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

CSV_PATH = "results/sweep_p500_runs.csv"

# Categorical slots 1-3 of the reference palette. These three are the set that
# validates under the all-pairs rule that scatter forms require.
NORM_COLOR = {
    "max": "#2a78d6",
    "frobenius": "#eb6834",
    "max_then_frobenius": "#1baf7a",
}
NORM_LABEL = {
    "max": "max",
    "frobenius": "Frobenius",
    "max_then_frobenius": "max → Frobenius",
}
NORM_ORDER = ["max", "frobenius", "max_then_frobenius"]

# Fixed left-to-right order so every setting's figure reads the same way.
ARM_ORDER = [
    "cocolasso_max", "cocolasso_frobenius",
    "reweighted_max", "reweighted_frobenius", "reweighted_max_then_frobenius",
    "cocolasso_refit_max", "cocolasso_refit_frobenius",
    "reweighted_refit_max", "reweighted_refit_frobenius",
    "reweighted_refit_max_then_frobenius",
]
ARM_TICK = {
    "cocolasso_max": "coco\nmax",
    "cocolasso_frobenius": "coco\nfrob",
    "reweighted_max": "rew\nmax",
    "reweighted_frobenius": "rew\nfrob",
    "reweighted_max_then_frobenius": "rew\nmax→frob",
    "cocolasso_refit_max": "coco-r\nmax",
    "cocolasso_refit_frobenius": "coco-r\nfrob",
    "reweighted_refit_max": "rew-r\nmax",
    "reweighted_refit_frobenius": "rew-r\nfrob",
    "reweighted_refit_max_then_frobenius": "rew-r\nmax→frob",
}

# (baseline arm, reweighted arm, panel title, same_norm). Same norm and same
# variant on both axes, so the reweighting is the only difference.
# max_then_frobenius has no entry here: CoCoLasso runs a single iteration, so
# that schedule degenerates to plain max and there is nothing to pair it with
# at the same norm. It appears in CROSS_PAIRS instead.
PAIRS = [
    ("cocolasso_max", "reweighted_max", "plain · max", True),
    ("cocolasso_frobenius", "reweighted_frobenius", "plain · Frobenius", True),
    ("cocolasso_refit_max", "reweighted_refit_max", "refit · max", True),
    ("cocolasso_refit_frobenius", "reweighted_refit_frobenius", "refit · Frobenius", True),
]

# The practical question: can the expensive projection be skipped altogether?
# CoCoLasso under the max norm is the published baseline and the costly one
# (~18.5 s/fit at p=500); the reweighted algorithm under Frobenius is ~5.5, and
# the hybrid ~23.9 against pure reweighted max at ~185. Here the norms differ
# deliberately -- that is the point of the comparison, not a caveat -- so these
# panels are not tagged cross-norm.
CROSS_PAIRS = [
    ("cocolasso_max", "reweighted_frobenius",
     "plain · rew Frobenius vs coco max", True),
    ("cocolasso_refit_max", "reweighted_refit_frobenius",
     "refit · rew Frobenius vs coco max", True),
    ("cocolasso_max", "reweighted_max_then_frobenius",
     "plain · rew max→frob vs coco max", True),
    ("cocolasso_refit_max", "reweighted_refit_max_then_frobenius",
     "refit · rew max→frob vs coco max", True),
]

GRID = dict(linestyle="-", linewidth=0.5, alpha=0.35)
INK = "#333333"


def norm_of(arm: str) -> str:
    for n in ("max_then_frobenius", "frobenius", "max"):
        if arm.endswith(n):
            return n
    return "max"


def setting_dir(root: Path, setting: str) -> Path:
    return root / setting


def plot_all_algorithms(df, setting, metric, out_path):
    """One column per algorithm, one point per repetition."""
    arms = [a for a in ARM_ORDER if a in set(df["arm"])]
    if not arms:
        return False

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    rng = np.random.default_rng(0)          # jitter only; fixed so reruns match

    for x, arm in enumerate(arms):
        vals = df.loc[df["arm"] == arm, metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        color = NORM_COLOR[norm_of(arm)]
        ax.scatter(x + rng.uniform(-0.16, 0.16, vals.size), vals,
                   s=9, color=color, alpha=0.35, linewidth=0, zorder=2)
        med = float(np.median(vals))
        ax.plot([x - 0.32, x + 0.32], [med, med], color=INK, linewidth=1.8, zorder=4)
        # Direct label: aqua sits under 3:1 contrast on a light surface, so the
        # value must be readable without relying on the colour. Offset to the
        # right of the median rule rather than above it, where the point cloud
        # is densest.
        ax.annotate(f"{med:.2f}", (x + 0.32, med), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8,
                    color=INK, fontweight="medium", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75))

    # Separate the plain block from the refit block. The labels sit just under
    # the top spine rather than above the axes, where they would collide with
    # the title.
    n_plain = sum(1 for a in arms if "_refit_" not in a)
    if 0 < n_plain < len(arms):
        ax.axvline(n_plain - 0.5, color="#bbbbbb", linewidth=1, linestyle=(0, (4, 3)))
        for xc, label in (((n_plain - 1) / 2, "plain"),
                          ((n_plain + len(arms) - 1) / 2, "refit on top-k")):
            ax.text(xc, 0.985, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color="#6b6a66")

    ax.set_yscale("log")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_TICK.get(a, a) for a in arms], fontsize=8)
    ax.set_ylabel(f"{metric.upper()} (log scale)", fontsize=10)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=NORM_COLOR[n],
                          markersize=7, label=NORM_LABEL[n])
               for n in NORM_ORDER if any(norm_of(a) == n for a in arms)]
    handles.append(plt.Line2D([], [], color=INK, linewidth=1.8, label="median"))
    # Below the axes: at 100 points per column there is no empty corner inside
    # the plot for a legend to sit in without covering data.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, fontsize=9, ncol=len(handles))

    n_reps = int(df.groupby("arm").size().max())
    ax.set_title(f"{setting} — {metric.upper()} by algorithm, one point per repetition "
                 f"({n_reps} reps)", fontsize=11, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_paired(df, setting, metric, out_path, pairs, xname, yname, subtitle):
    """One point per shared seed, `xname` on x against `yname` on y.

    Used for both comparisons: CoCoLasso against the reweighted algorithm at a
    fixed norm, and the max norm against Frobenius at a fixed algorithm. In
    both cases the two arms ran on identical data for a given seed, so the
    pairing is exact and the y=x diagonal is the whole reading.
    """
    wide = df.pivot_table(index="seed", columns="arm", values=metric, aggfunc="first")
    panels = [(b, r, t, same) for b, r, t, same in pairs
              if b in wide.columns and r in wide.columns
              and wide[[b, r]].dropna().shape[0] > 0]
    if not panels:
        return False

    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.3 * nrow), dpi=150,
                             squeeze=False)

    for ax, (base, rew, title, same_norm) in zip(axes.ravel(), panels):
        sub = wide[[base, rew]].dropna()
        x = sub[base].to_numpy(dtype=float)
        y = sub[rew].to_numpy(dtype=float)
        keep = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if x.size == 0:
            ax.set_visible(False)
            continue

        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = (hi / lo) ** 0.05 if lo > 0 else 1.1
        lims = (lo / pad, hi * pad)
        ax.plot(lims, lims, color="#999999", linewidth=1,
                linestyle=(0, (4, 3)), zorder=1)

        color = NORM_COLOR[norm_of(rew)]
        ax.scatter(x, y, s=16, color=color, alpha=0.5, linewidth=0, zorder=3)

        wins = int((y < x).sum())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{xname} {metric.upper()}", fontsize=9)
        ax.set_ylabel(f"{yname} {metric.upper()}", fontsize=9)
        head = title if same_norm else title + "  (cross-norm)"
        ax.set_title(f"{head}\n{wins}/{x.size} seeds below the line "
                     f"({100*wins/x.size:.0f}% {yname} wins)",
                     fontsize=9.5, fontweight="bold", loc="left")
        # On a narrow log range matplotlib labels many minor steps in scientific
        # notation ("1.2 x 10^1"), which collides at this panel width. Plain
        # numbers on a sparse 1/2/5 subdivision stay short and readable.
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0),
                                                      numticks=12))
            axis.set_minor_locator(mticker.LogLocator(base=10,
                                                      subs=(3.0, 4.0, 6.0, 8.0),
                                                      numticks=12))
            fmt = mticker.ScalarFormatter()
            fmt.set_scientific(False)
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis="both", which="major", labelsize=7.5)
        ax.grid(**GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(f"{setting} — {subtitle}\n"
                 f"below the dashed y=x line, {yname} won that dataset",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--metric", default="pe", choices=["pe", "mse", "time", "nnz"])
    ap.add_argument("--setting", default=None, help="only this setting")
    ap.add_argument("--outdir", default=None,
                    help="write every figure here instead of into each setting's "
                         "own results directory")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    root = Path(args.csv).parent / Path(args.csv).stem.replace("_runs", "")

    settings = sorted(df["setting"].unique())
    if args.setting:
        settings = [s for s in settings if s == args.setting]
        if not settings:
            print(f"no setting named {args.setting!r}")
            return 2

    n_all = n_algo = n_cross = 0
    for setting in settings:
        sub = df[df["setting"] == setting]
        out_dir = Path(args.outdir) if args.outdir else setting_dir(root, setting)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{setting}_" if args.outdir else ""

        if plot_all_algorithms(sub, setting, args.metric,
                               out_dir / f"{stem}compare_all_{args.metric}.png"):
            n_all += 1
        # Same data, two cuts: hold the norm fixed and vary the algorithm, then
        # hold the algorithm fixed and vary the norm.
        if plot_paired(sub, setting, args.metric,
                       out_dir / f"{stem}paired_coco_vs_rew_{args.metric}.png",
                       PAIRS, "CoCoLasso", "reweighted",
                       "CoCoLasso vs reweighted, paired by seed "
                       "(same norm, same repetition, identical data)"):
            n_algo += 1
        if plot_paired(sub, setting, args.metric,
                       out_dir / f"{stem}paired_rew_cheap_vs_coco_max_{args.metric}.png",
                       CROSS_PAIRS, "CoCoLasso · max", "reweighted",
                       "reweighted on a cheap norm vs CoCoLasso on the max norm, "
                       "paired by seed (same repetition, identical data)"):
            n_cross += 1

    print(f"{len(settings)} settings: {n_all} all-algorithm, {n_algo} coco-vs-rew, "
          f"{n_cross} cheap-vs-coco-max figures ({args.metric})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
