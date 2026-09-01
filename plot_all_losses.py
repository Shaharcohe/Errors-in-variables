"""One scatter of every algorithm's loss, over runs matched across all of them.

    python plot_all_losses.py
    python plot_all_losses.py --metric mse
    python plot_all_losses.py --exclude reweighted_max reweighted_refit_max

Only runs whose (setting, seed) exists for *every* algorithm plotted are shown.
The intersection is taken first and reported before anything is drawn, so the
columns are strictly comparable: each algorithm is scored on exactly the same
datasets as every other, and no arm is flattered by having been run on an
easier or merely different subset. An arm still mid-sweep therefore shrinks the
whole figure, which is why --exclude exists.

Encoding matches paired_diff.py so a point means the same thing in both:
shape = noise level, colour = sparsity. The algorithm is the x position; x has
no quantitative meaning and carries jitter only to separate overlapping points.

Nothing is averaged -- every matched repetition is its own point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paired_diff import (NOISE_LEVEL, NOISE_ORDER, NOISE_MARKER,
                         SPARSITY_COLOR, SPARSITY_ORDER, sparsity_label,
                         GRID, INK)

CSV_IN = "results/sweep_p500_runs.csv"

# Left-to-right order, and the two-line tick label for each.
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
ARM_KEY = [("coco", "CoCoLasso"), ("rew", "reweighted (yours)"),
           ("-r", "+ refit on top-k"),
           ("max / frob", "projection norm"),
           ("max→frob", "max at t=0, Frobenius after")]

METRIC_LABEL = {"pe": "PE (prediction error)", "mse": "MSE",
                "time": "seconds per fit"}


def matched(df: pd.DataFrame, arms: list[str]):
    """Intersect (setting, seed) across every arm, reporting as it goes."""
    print(f"algorithms compared ({len(arms)}): {', '.join(arms)}\n")
    keys = {}
    for a in arms:
        sub = df[df["arm"] == a]
        keys[a] = set(map(tuple, sub[["setting", "seed"]].to_numpy()))
        print(f"  {a:38} {len(sub):5} runs  "
              f"{sub['setting'].nunique():2} settings")

    inter = set.intersection(*keys.values())
    print(f"\nintersection of (setting, seed) across all "
          f"{len(arms)} algorithms: {len(inter)}")
    if not inter:
        return pd.DataFrame(), inter

    settings = sorted({s for s, _ in inter})
    print(f"  {len(settings)} settings x {len(inter) // max(len(settings), 1)} "
          f"seeds (expected 100 per setting)")

    # Name the arms that are actually costing coverage, so a small figure is
    # explained rather than mysterious.
    full = max(len(k) for k in keys.values())
    binding = [a for a, k in keys.items() if len(k) < full]
    if binding:
        print("  limited by: " + ", ".join(
            f"{a} ({len(keys[a])})" for a in sorted(binding, key=lambda a: len(keys[a]))))

    idx = pd.MultiIndex.from_tuples(sorted(inter), names=["setting", "seed"])
    out = df[df["arm"].isin(arms)].set_index(["setting", "seed"])
    out = out[out.index.isin(idx)].reset_index()
    print(f"  plotted points: {len(out)} "
          f"({len(inter)} matched runs x {len(arms)} algorithms)")
    return out, inter


def plot(tab, arms, metric, out_path, logy=True):
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    rng = np.random.default_rng(0)
    noises = [n for n in NOISE_ORDER if n in set(tab["noise_level"])]
    sparsities = [s for s in SPARSITY_ORDER if s in set(tab["sparsity"])]

    for x, arm in enumerate(arms):
        sub = tab[tab["arm"] == arm]
        if sub.empty:
            continue
        for noise in noises:
            s2 = sub[sub["noise_level"] == noise]
            if s2.empty:
                continue
            ax.scatter(x + rng.uniform(-0.34, 0.34, len(s2)), s2[metric],
                       marker=NOISE_MARKER[noise], s=11,
                       c=[SPARSITY_COLOR[s] for s in s2["sparsity"]],
                       alpha=0.5, linewidth=0, zorder=3)
        med = float(np.median(sub[metric]))
        ax.plot([x - 0.4, x + 0.4], [med, med], color=INK, linewidth=2, zorder=5)
        ax.annotate(f"{med:.2f}", (x + 0.4, med), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.5, color=INK, fontweight="medium", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75))

    n_plain = sum(1 for a in arms if "_refit_" not in a)
    if 0 < n_plain < len(arms):
        ax.axvline(n_plain - 0.5, color="#bbbbbb", linewidth=1,
                   linestyle=(0, (4, 3)), zorder=1)
        for xc, lab in (((n_plain - 1) / 2, "plain"),
                        ((n_plain + len(arms) - 1) / 2, "refit on top-k")):
            ax.text(xc, 0.985, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color="#6b6a66")

    if logy:
        ax.set_yscale("log")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_TICK.get(a, a) for a in arms], fontsize=8)
    ax.set_xlim(-0.7, len(arms) - 0.3)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric.upper())
                  + (" (log scale)" if logy else ""), fontsize=10)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    inv = {v: k for k, v in NOISE_LEVEL.items()}
    h_noise = [plt.Line2D([], [], marker=NOISE_MARKER[n], linestyle="", color=INK,
                          markersize=6, label=f"{n} (σₐ={inv[n]})") for n in noises]
    h_sparse = [plt.Line2D([], [], marker="o", linestyle="", color=SPARSITY_COLOR[s],
                           markersize=7, label=s) for s in sparsities]
    h_arm = [plt.Line2D([], [], linestyle="", marker="", label=f"{k} = {v}")
             for k, v in ARM_KEY]
    h_arm.append(plt.Line2D([], [], color=INK, linewidth=2, label="median"))

    l1 = ax.legend(handles=h_noise, title="noise level", frameon=False,
                   loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   fontsize=8, title_fontsize=8.5)
    ax.add_artist(l1)
    l2 = ax.legend(handles=h_sparse, title="sparsity", frameon=False,
                   loc="upper left", bbox_to_anchor=(1.01, 0.78),
                   fontsize=8, title_fontsize=8.5)
    ax.add_artist(l2)
    ax.legend(handles=h_arm, title="algorithm (x axis)", frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 0.47),
              fontsize=8, title_fontsize=8.5, handlelength=0, handletextpad=0)

    n_match = len(tab) // max(len(arms), 1)
    ax.set_title(f"{METRIC_LABEL.get(metric, metric.upper())} by algorithm",
                 fontsize=12, fontweight="bold", loc="left", pad=22)
    ax.text(0.0, 1.015, f"{n_match} runs matched on (setting, seed) across all "
                        f"{len(arms)} algorithms · {len(tab)} points · "
                        f"{tab['setting'].nunique()} settings",
            transform=ax.transAxes, fontsize=8.5, color="#6b6a66", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_IN)
    ap.add_argument("--metric", default="pe", choices=["pe", "mse", "time"])
    # Both pure-max reweighted arms are excluded by default: they are the
    # chunked ~77-minute jobs and are still finishing, and including either one
    # shrinks the matched intersection from 2200 runs across 22 settings to
    # 1325 across 18. Drop the flag once the sweep completes.
    ap.add_argument("--exclude", nargs="*",
                    default=["reweighted_max", "reweighted_refit_max"],
                    help="algorithms to leave out of the comparison entirely "
                         "[default: the two pure-max reweighted arms]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--linear", action="store_true", help="linear y axis")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    df["noise_level"] = df["sigma_a"].map(NOISE_LEVEL)
    df["sparsity"] = df.apply(sparsity_label, axis=1)

    arms = [a for a in ARM_ORDER
            if a in set(df["arm"]) and a not in set(args.exclude)]
    if args.exclude:
        print(f"excluded: {', '.join(args.exclude)}\n")
    if not arms:
        print("no algorithms left to plot", file=sys.stderr)
        return 2

    tab, inter = matched(df, arms)
    if tab.empty:
        print("\nno (setting, seed) is present for every algorithm -- nothing "
              "to plot. Exclude whichever arm is still running.", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else Path(
        f"results/all_algorithms_{args.metric}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(tab, arms, args.metric, out, logy=not args.linear)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
