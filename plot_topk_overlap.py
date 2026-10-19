"""Mean top-10 overlaps from topk_overlap.py, one figure per metric.

    python topk_overlap.py && python plot_topk_overlap.py

Grid: rows = covariance model (AR, CS), columns = beta* preset; x = tau
(sigma_a). Each arm is a dodged point with a +/- 1 se bar over its 100 seeds.
Colour = projection norm (as in plot_settings.py), marker = algorithm (hollow
diamond CoCoLasso, filled circle ours, filled triangle two-step), so the arm
never rests on colour alone. Every arm present in the CSV is drawn.

Writes results/topk_overlap_top{10,3}.png (means) and *_dots.png (per seed).
"""
from __future__ import annotations

import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# As in paired_diff.py / plot_settings.py (not imported: those need pandas).
GRID = dict(linestyle="-", linewidth=0.5, alpha=0.35)
INK = "#333333"
NORM_COLOR = {"max": "#2a78d6", "frobenius": "#eb6834"}

CSV_IN = "results/topk_overlap_runs.csv"
ALGO_ORDER = ["cocolasso", "reweighted", "twostep"]
NORM_LABEL = {"max": "max", "frobenius": "Frobenius"}


def arm_order(arm):
    algo = arm.split("_", 1)[0]
    return (ALGO_ORDER.index(algo), arm, not arm.endswith("max"))


def arm_label(arm):
    norm = NORM_LABEL["max" if arm.endswith("max") else "frobenius"]
    if arm.startswith("cocolasso"):
        return f"CoCoLasso · {norm}"
    if arm.startswith("reweighted"):
        return f"ours · {norm}"
    gamma = arm.split("_")[1].removeprefix("g")        # twostep_g<gamma>_<norm>
    return f"two-step γ={gamma} · {norm}"
BETAS = ["datta_zhang", "weak_sparse_a1.0", "weak_sparse_a1.5", "weak_sparse_a2.0"]
BETA_TITLE = {"datta_zhang": "β* = (3, 1.5, 0, 0, 2, 0, …)",
              "weak_sparse_a1.0": "weakly sparse, η = 1.0",
              "weak_sparse_a1.5": "weakly sparse, η = 1.5",
              "weak_sparse_a2.0": "weakly sparse, η = 2.0"}
MODELS = {"ar": "AR", "cs": "CS"}
TAUS = [0.75, 1.0, 1.25]
METRICS = {
    "top10": ("top10_vs_top10",
              r"$|\mathrm{top}_{10}(\hat\beta) \cap \mathrm{top}_{10}(\beta^*)|$",
              "Overlap of the estimate's top 10 with the true top 10 (max 10)"),
    "top3": ("top10_vs_top3",
             r"$|\mathrm{top}_{10}(\hat\beta) \cap \mathrm{top}_{3}(\beta^*)|$",
             "Overlap of the estimate's top 10 with the true top 3 (max 3)"),
}
# One y range for every panel and both figures, so settings compare directly
# and arm differences are shown at their true size; every mean is below 3.
YLIM = (0, 3.1)
DODGE = 0.075


def style(arm):
    color = NORM_COLOR["max" if arm.endswith("max") else "frobenius"]
    coco = arm.startswith("cocolasso")
    marker = "D" if coco else "^" if arm.startswith("twostep") else "o"
    return dict(color=color, marker=marker, markersize=6.5,
                markerfacecolor="white" if coco else color,
                markeredgecolor=color, markeredgewidth=1.6)


def plot(rows, key, arms):
    col, ylabel, title = METRICS[key]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["model"], r["beta"], float(r["sigma_a"]), r["arm"])].append(float(r[col]))
    mean = {g: np.mean(v) for g, v in groups.items()}
    se = {g: np.std(v, ddof=1) / np.sqrt(len(v)) for g, v in groups.items()}

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, sharey=True)
    axes[0, 0].set_ylim(*YLIM)
    for i, model in enumerate(MODELS):
        for j, beta in enumerate(BETAS):
            ax = axes[i, j]
            for a, arm in enumerate(arms):
                xs = np.arange(len(TAUS)) + (a - (len(arms) - 1) / 2) * DODGE
                m = [mean[(model, beta, t, arm)] for t in TAUS]
                s = [se[(model, beta, t, arm)] for t in TAUS]
                ax.errorbar(xs, m, yerr=s, linestyle="", capsize=0,
                            elinewidth=1.2, ecolor=style(arm)["color"],
                            label=arm_label(arm), zorder=3, **style(arm))
            ax.grid(axis="y", **GRID)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xticks(range(len(TAUS)))
            ax.set_xticklabels([f"{t:g}" for t in TAUS])
            ax.set_xlim(-0.45, len(TAUS) - 0.55)
            if i == 0:
                ax.set_title(BETA_TITLE[beta], fontsize=10, color=INK)
            if i == 1:
                ax.set_xlabel("τ (measurement-error sd)", color=INK)
            if j == 0:
                ax.set_ylabel(f"{MODELS[model]}\n{ylabel}", color=INK)
    if key == "top10":
        axes[0, 0].annotate("β* has 3 nonzeros: true top-10 = support",
                            xy=(0.02, 0.03), xycoords="axes fraction",
                            fontsize=8, color="#6b6a66")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(arms), 4),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{title}  (mean ± 1 se over 100 seeds; n=100, p=500, λ=1)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04 + 0.03 * ((len(arms) - 1) // 4), 1, 1))
    out = f"results/topk_overlap_{key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


DOT_WIDTH = 0.8         # x span of all arms at one tau
DOT_JITTER_Y = 0.28     # half-width of the y jitter


def plot_dots(rows, key, arms):
    """Every seed as a dot (jittered in x and y around its integer value)."""
    col, ylabel, title = METRICS[key]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["model"], r["beta"], float(r["sigma_a"]), r["arm"])].append(float(r[col]))
    vmax = max(max(v) for v in groups.values())
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for i, model in enumerate(MODELS):
        for j, beta in enumerate(BETAS):
            ax = axes[i, j]
            slot = DOT_WIDTH / len(arms)
            for a, arm in enumerate(arms):
                st = style(arm)
                marker = "^" if st["marker"] == "^" else "o"
                for x0, tau in enumerate(TAUS):
                    v = np.array(groups[(model, beta, tau, arm)])
                    x = x0 + (a - (len(arms) - 1) / 2) * slot
                    ax.scatter(x + rng.uniform(-1, 1, v.size) * 0.35 * slot,
                               v + rng.uniform(-1, 1, v.size) * DOT_JITTER_Y,
                               s=7, marker=marker, facecolor=st["markerfacecolor"],
                               edgecolor=st["color"], linewidth=0.6, alpha=0.6,
                               zorder=2)
                    ax.plot([x - 0.45 * slot, x + 0.45 * slot], [v.mean()] * 2,
                            color=INK, lw=1.8, zorder=4)
            ax.grid(axis="y", **GRID)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xticks(range(len(TAUS)))
            ax.set_xticklabels([f"{t:g}" for t in TAUS])
            ax.set_xlim(-0.5, len(TAUS) - 0.5)
            ax.set_ylim(-0.5, vmax + 0.5)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
            if i == 0:
                ax.set_title(BETA_TITLE[beta], fontsize=10, color=INK)
            if i == 1:
                ax.set_xlabel("τ (measurement-error sd)", color=INK)
            if j == 0:
                ax.set_ylabel(f"{MODELS[model]}\n{ylabel}", color=INK)

    # Proxies: resizing the real collections would also resize the plotted dots.
    handles = [plt.Line2D([], [], ls="", markersize=5.5,
                          marker="^" if style(a)["marker"] == "^" else "o",
                          markerfacecolor=style(a)["markerfacecolor"],
                          markeredgecolor=style(a)["color"]) for a in arms]
    handles.append(plt.Line2D([], [], color=INK, lw=1.8))
    labels = [arm_label(a) for a in arms] + ["mean"]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{title}: one dot per seed, 100 seeds per arm "
                 f"(y jittered ±{DOT_JITTER_Y} around the integer value)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04 + 0.03 * (len(arms) // 4), 1, 1))
    out = f"results/topk_overlap_{key}_dots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    rows = list(csv.DictReader(open(CSV_IN)))
    arms = sorted({r["arm"] for r in rows}, key=arm_order)
    for key in METRICS:
        plot(rows, key, arms)
        plot_dots(rows, key, arms)


if __name__ == "__main__":
    main()
