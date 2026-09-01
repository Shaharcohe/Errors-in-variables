"""Paired difference between CoCoLasso (max norm) and the reweighted algorithm
(Frobenius norm), one row and one point per repetition.

    python paired_diff.py
    python paired_diff.py --symlog        # if breakdowns compress the bulk

Writes:

    results/paired_coco_max_vs_rew_frob.csv     one row per paired repetition
    results/paired_diff_main.png                all pairs, shape=noise, colour=sparsity
    results/paired_diff_by_noise.png            grouped by noise level
    results/paired_diff_by_sparsity.png         grouped by sparsity level

Sign convention, identical in the CSV and in every figure:

    diff = metric(CoCoLasso · max) - metric(reweighted · Frobenius)

so **positive means the reweighted algorithm did better** on that dataset, and
y = 0 is the no-difference line.

The pairing is exact rather than approximate. Data generation is deterministic
given (seed, model, n, p, sigma_a, sigma_e, beta*), and every arm was run at
seeds 0..99, so two rows sharing a (setting, seed) were fit on bit-identical
data. verify_pairs() re-checks that claim against the CSV columns rather than
assuming it, and refuses to pair rows whose experimental parameters disagree.

Nothing is averaged: each repetition keeps its own row and its own point.
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

CSV_IN = "results/sweep_p500_runs.csv"
CSV_OUT = "results/paired_coco_max_vs_rew_frob.csv"

ARM_A = "cocolasso_max"          # baseline, the x/left side of the difference
ARM_B = "reweighted_frobenius"   # the reweighted algorithm

YLABEL = "PE(CoCoLasso · max) − PE(reweighted · Frobenius)"

# metric key -> (column, y label, reference line, log y)
#
# pe/mse/time keep one sign convention: baseline minus reweighted, so positive
# always means the reweighted algorithm did better (lower error, less time).
#
# speedup is a ratio rather than a difference. Runtime spans a multiple, not an
# offset -- a 3x speedup means the same thing at 5 s and at 200 s, while "-13
# seconds" does not -- so the ratio on a log axis is the honest encoding, and
# its no-difference line is 1 rather than 0.
METRICS = {
    "pe": ("pe_diff", YLABEL, 0.0, False),
    "mse": ("mse_diff",
            "MSE(CoCoLasso · max) − MSE(reweighted · Frobenius)", 0.0, False),
    "time": ("time_diff",
             "seconds(CoCoLasso · max) − seconds(reweighted · Frobenius)",
             0.0, False),
    "speedup": ("speedup",
                "runtime ratio  CoCoLasso · max  /  reweighted · Frobenius",
                1.0, True),
}

# Parameters that must agree between the two rows of a pair. Anything that
# differs here means the two runs are not comparable and the pair is dropped.
#
# snr is compared with a tolerance, not exactly. It is *derived* -- config.snr
# recomputes beta*' Sigma_x beta* / sigma_e^2 inside each job -- and identical
# inputs on two different compute nodes can land a few ulps apart. Its actual
# inputs (model, p, beta_preset, alpha, sigma_e) are all checked exactly above
# it, so an exact check on snr adds nothing and silently discards good pairs.
MUST_MATCH = ["model", "n", "p", "sigma_a", "sigma_e", "beta_preset", "alpha",
              "lam_requested", "k", "true_k", "snr"]
APPROX_MATCH = {"snr"}

# Noise level -> marker. Three shapes, so identity never rests on colour alone.
NOISE_LEVEL = {0.75: "low", 1.0: "medium", 1.25: "high"}
NOISE_ORDER = ["low", "medium", "high"]
NOISE_MARKER = {"low": "o", "medium": "^", "high": "s"}

# Four distinct hues rather than one ramp, so the levels separate at a glance
# in a dense cloud. Slots 1/2/3/7 of the categorical palette: blue, orange,
# aqua, violet. The 4th slot (yellow) is deliberately skipped -- it is the
# documented failing pair against orange -- as are red and magenta, which sit
# too close to orange and too low in contrast respectively.
#
# Caveat worth knowing: the palette validates all-pairs colourblind separation
# for three categorical hues, not four, and node is unavailable here to re-run
# the checker. Noise level is on marker shape, so that channel is safe, but
# sparsity now rests on colour alone. Facet by sparsity if a CVD-safe figure is
# needed.
SPARSITY_COLOR = {
    "weak α=1.0 (83% in first 3)": "#2a78d6",
    "weak α=1.5 (97% in first 3)": "#eb6834",
    "weak α=2.0 (99% in first 3)": "#1baf7a",
    "datta_zhang (exactly 3-sparse)": "#4a3aa7",
}
SPARSITY_ORDER = list(SPARSITY_COLOR)

GRID = dict(linestyle="-", linewidth=0.5, alpha=0.35)
INK = "#333333"


def sparsity_label(row) -> str:
    if row["beta_preset"] == "datta_zhang":
        return "datta_zhang (exactly 3-sparse)"
    pct = {1.0: 83, 1.5: 97, 2.0: 99}.get(float(row["alpha"]))
    tail = f" ({pct}% in first 3)" if pct else ""
    return f"weak α={float(row['alpha']):.1f}{tail}"


def verify_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Pair the two arms on (setting, seed) and check every pair is legitimate.

    Reports what was found and what was refused, so a partially-finished sweep
    cannot quietly produce a half-populated figure that looks complete.
    """
    a = df[df["arm"] == ARM_A]
    b = df[df["arm"] == ARM_B]
    print(f"rows: {len(a)} {ARM_A}, {len(b)} {ARM_B}")
    if a.empty or b.empty:
        print("  ERROR: one of the two arms has no rows at all", file=sys.stderr)
        return pd.DataFrame()

    for name, arm in ((ARM_A, a), (ARM_B, b)):
        dup = arm.duplicated(["setting", "seed"]).sum()
        if dup:
            print(f"  WARNING: {dup} duplicate (setting, seed) rows in {name}",
                  file=sys.stderr)

    merged = a.merge(b, on=["setting", "seed"], suffixes=("_a", "_b"), how="inner")
    print(f"candidate pairs on (setting, seed): {len(merged)}")

    # Every experimental parameter must agree; the arms may differ only in
    # algorithm and norm.
    ok = pd.Series(True, index=merged.index)
    for col in MUST_MATCH:
        ca, cb = f"{col}_a", f"{col}_b"
        if ca not in merged or cb not in merged:
            continue
        both_nan = merged[ca].isna() & merged[cb].isna()
        if col in APPROX_MATCH:
            same = both_nan | np.isclose(pd.to_numeric(merged[ca], errors="coerce"),
                                         pd.to_numeric(merged[cb], errors="coerce"),
                                         rtol=1e-9, atol=0.0, equal_nan=True)
        else:
            same = (merged[ca] == merged[cb]) | both_nan
        if not same.all():
            print(f"  WARNING: {int((~same).sum())} pairs disagree on {col!r} "
                  f"-- dropped", file=sys.stderr)
        ok &= same

    # Both metrics must actually be present.
    for col in ("pe", "mse"):
        finite = np.isfinite(merged[f"{col}_a"]) & np.isfinite(merged[f"{col}_b"])
        if not finite.all():
            print(f"  WARNING: {int((~finite).sum())} pairs missing a {col} value "
                  f"-- dropped", file=sys.stderr)
        ok &= finite

    merged = merged[ok]

    # The repetition index must line up too: identical seed AND identical
    # position in the run. They can differ when an arm was chunked, so this is
    # reported rather than enforced -- the seed is what identifies the dataset.
    if "rep_index_a" in merged and "rep_index_b" in merged:
        off = int((merged["rep_index_a"] != merged["rep_index_b"]).sum())
        if off:
            print(f"  note: {off} pairs have different rep_index but the same seed "
                  f"(expected when one arm was chunked; the seed identifies the "
                  f"dataset, so the pairing still holds)")

    print(f"VALID PAIRS: {len(merged)}")
    if len(merged):
        per_setting = merged.groupby("setting").size()
        print(f"  across {per_setting.size} settings, "
              f"{per_setting.min()}-{per_setting.max()} pairs each "
              f"(expected 100)")
        short = per_setting[per_setting < 100]
        if len(short):
            print(f"  incomplete settings ({len(short)}): "
                  f"{', '.join(f'{s}={n}' for s, n in short.items())}")
    return merged


def build_table(merged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "setting": merged["setting"],
        "seed": merged["seed"],
        "rep_index": merged["rep_index_a"],
        "model": merged["model_a"],
        "n": merged["n_a"],
        "p": merged["p_a"],
        "sigma_a": merged["sigma_a_a"],
        "sigma_e": merged["sigma_e_a"],
        "beta_preset": merged["beta_preset_a"],
        "alpha": merged["alpha_a"],
        "k": merged["k_a"],
        "true_k": merged["true_k_a"],
        "lam": merged["lam_requested_a"],
        "snr": merged["snr_a"],
        "algo_a": ARM_A,
        "algo_b": ARM_B,
        "mse_cocolasso_max": merged["mse_a"],
        "mse_reweighted_frobenius": merged["mse_b"],
        "pe_cocolasso_max": merged["pe_a"],
        "pe_reweighted_frobenius": merged["pe_b"],
        "time_cocolasso_max": merged["time_a"],
        "time_reweighted_frobenius": merged["time_b"],
    })
    # Sign convention: positive => the reweighted algorithm did better.
    out["mse_diff"] = out["mse_cocolasso_max"] - out["mse_reweighted_frobenius"]
    out["pe_diff"] = out["pe_cocolasso_max"] - out["pe_reweighted_frobenius"]
    out["time_diff"] = (out["time_cocolasso_max"]
                        - out["time_reweighted_frobenius"])
    # Ratio > 1 means the reweighted algorithm was that many times faster.
    out["speedup"] = (out["time_cocolasso_max"]
                      / out["time_reweighted_frobenius"].replace(0, np.nan))
    out["noise_level"] = out["sigma_a"].map(NOISE_LEVEL)
    out["sparsity"] = out.apply(sparsity_label, axis=1)
    return out.sort_values(["setting", "seed"]).reset_index(drop=True)


def _finish(ax, tab, symlog, metric):
    col, ylabel, ref, logy = METRICS[metric]
    ax.axhline(ref, color=INK, linewidth=1.2, linestyle=(0, (5, 4)), zorder=4)
    if logy:
        ax.set_yscale("log")
    elif symlog:
        # Keeps a linear window around zero so the sign stays readable while a
        # blown-up repetition does not flatten the bulk of the points.
        span = float(np.nanpercentile(np.abs(tab[col]), 90)) or 1.0
        ax.set_yscale("symlog", linthresh=max(span, 1e-6))
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _legends(ax, tab, noises, sparsities):
    """Both legends outside the axes -- a dense point cloud leaves no free
    corner inside the plot for either of them to sit in."""
    inv = {v: k for k, v in NOISE_LEVEL.items()}
    shape_h = [plt.Line2D([], [], marker=NOISE_MARKER[n], linestyle="", color=INK,
                          markersize=6, label=f"{n} (σₐ={inv[n]})")
               for n in noises]
    color_h = [plt.Line2D([], [], marker="o", linestyle="", color=SPARSITY_COLOR[s],
                          markersize=7, label=s) for s in sparsities]
    l1 = ax.legend(handles=shape_h, title="noise level", frameon=False,
                   loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   fontsize=8, title_fontsize=8.5)
    ax.add_artist(l1)
    ax.legend(handles=color_h, title="sparsity", frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 0.72),
              fontsize=8, title_fontsize=8.5)


def plot_main(tab, out_path, symlog, metric="pe"):
    """Every pair in one column, jittered; shape = noise, colour = sparsity."""
    col, _ylabel, ref, _logy = METRICS[metric]
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=150)
    rng = np.random.default_rng(0)
    noises = [n for n in NOISE_ORDER if n in set(tab["noise_level"])]
    sparsities = [s for s in SPARSITY_ORDER if s in set(tab["sparsity"])]

    for noise in noises:
        for sp in sparsities:
            sub = tab[(tab["noise_level"] == noise) & (tab["sparsity"] == sp)]
            if sub.empty:
                continue
            ax.scatter(rng.uniform(-0.42, 0.42, len(sub)), sub[col],
                       marker=NOISE_MARKER[noise], s=16,
                       color=SPARSITY_COLOR[sp], alpha=0.55, linewidth=0, zorder=3)

    wins = int((tab[col] > ref).sum())
    ax.set_xlim(-0.75, 0.75)
    ax.set_xticks([])
    _finish(ax, tab, symlog, metric)
    _legends(ax, tab, noises, sparsities)
    what = "Runtime ratio" if metric == "speedup" else f"{metric.upper()} difference"
    ax.set_title(f"{what} per paired repetition",
                 fontsize=12, fontweight="bold", loc="left", pad=22)
    # The counts belong on the figure but not in its title.
    verb = "faster" if metric in ("time", "speedup") else "better"
    ax.text(0.0, 1.015, f"{len(tab)} pairs · reweighted {verb} in "
                        f"{wins} ({100*wins/len(tab):.0f}%)",
            transform=ax.transAxes, fontsize=8.5, color="#6b6a66", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_grouped(tab, column, order, out_path, symlog, xlabel, metric="pe"):
    """One jittered column per group, still one point per repetition."""
    col, _ylabel, ref, _logy = METRICS[metric]
    groups = [g for g in order if g in set(tab[column])]
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(groups)), 5.8), dpi=150)
    rng = np.random.default_rng(0)

    for x, g in enumerate(groups):
        sub = tab[tab[column] == g]
        if sub.empty:
            continue
        # Keep both encodings in the grouped views so a point means the same
        # thing in every figure.
        for noise in [n for n in NOISE_ORDER if n in set(sub["noise_level"])]:
            s2 = sub[sub["noise_level"] == noise]
            colors = [SPARSITY_COLOR[s] for s in s2["sparsity"]]
            ax.scatter(x + rng.uniform(-0.3, 0.3, len(s2)), s2[col],
                       marker=NOISE_MARKER[noise], s=15, c=colors,
                       alpha=0.55, linewidth=0, zorder=3)
        med = float(np.median(sub[col]))
        ax.plot([x - 0.38, x + 0.38], [med, med], color=INK, linewidth=1.8, zorder=5)
        wins = int((sub[col] > ref).sum())
        shown = f"{med:.2f}x" if metric == "speedup" else f"{med:+.2f}"
        ax.annotate(f"median {shown}\n{100*wins/len(sub):.0f}%  (n={len(sub)})",
                    (x, med), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.8))

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace(" (", "\n(") for g in groups], fontsize=8.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xlabel(xlabel, fontsize=9.5)
    _finish(ax, tab, symlog, metric)
    wins = int((tab[col] > ref).sum())
    what = "Runtime ratio" if metric == "speedup" else f"{metric.upper()} difference"
    ax.set_title(f"{what} by {xlabel}",
                 fontsize=12, fontweight="bold", loc="left", pad=22)
    ax.text(0.0, 1.015, f"{len(tab)} pairs · {100*wins/len(tab):.0f}% overall",
            transform=ax.transAxes, fontsize=8.5, color="#6b6a66", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_IN)
    ap.add_argument("--out-csv", default=CSV_OUT)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--symlog", action="store_true",
                    help="symlog y axis, for when a breakdown flattens the bulk")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    merged = verify_pairs(df)
    if merged.empty:
        print("no valid pairs -- nothing written", file=sys.stderr)
        return 1

    tab = build_table(merged)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_csv, index=False)
    print(f"\nwrote {len(tab)} rows to {out_csv}")

    pe, mse = tab["pe_diff"], tab["mse_diff"]
    print(f"  PE  diff: median {pe.median():+.4f}  mean {pe.mean():+.4f}  "
          f"[{pe.min():+.3f}, {pe.max():+.3f}]  positive {int((pe > 0).sum())}/{len(pe)}")
    print(f"  MSE diff: median {mse.median():+.4f}  mean {mse.mean():+.4f}  "
          f"[{mse.min():+.3f}, {mse.max():+.3f}]  positive {int((mse > 0).sum())}/{len(mse)}")

    t_a, t_b = tab["time_cocolasso_max"], tab["time_reweighted_frobenius"]
    sp = tab["speedup"]
    print(f"  runtime : CoCoLasso·max median {t_a.median():.2f}s, "
          f"reweighted·Frobenius median {t_b.median():.2f}s")
    print(f"  speedup : median {sp.median():.2f}x  "
          f"[{sp.min():.2f}x, {sp.max():.2f}x]  "
          f"faster in {int((sp > 1).sum())}/{len(sp)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_main(tab, outdir / "paired_diff_main.png", args.symlog, "pe")
    plot_grouped(tab, "noise_level", NOISE_ORDER,
                 outdir / "paired_diff_by_noise.png", args.symlog,
                 "noise level", "pe")
    plot_grouped(tab, "sparsity", SPARSITY_ORDER,
                 outdir / "paired_diff_by_sparsity.png", args.symlog,
                 "sparsity", "pe")
    # One runtime figure, as a ratio: runtime spans a multiple rather than an
    # offset, so "4.6x faster" carries the meaning that "-13 seconds" does not.
    plot_main(tab, outdir / "paired_diff_runtime.png", args.symlog, "speedup")
    print(f"wrote 4 figures to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
