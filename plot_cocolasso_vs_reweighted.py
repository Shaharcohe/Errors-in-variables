"""Direct cocolasso-vs-reweighted comparison, one panel per projection norm.

Three bars per sigma_a: cocolasso, reweighted at its default rho=0.5, and
reweighted at rho=0. rho=0 is not excluded anymore, but where it breaks down
(median PE far outside the single-digit range everything else lives in) its
bar is replaced with a "failed" label rather than plotted -- an 8-11 order of
magnitude value would force a log axis that makes the real, single-digit
comparison between cocolasso and reweighted(rho=0.5) unreadable. See
plot_median_pe.py for the full sweep on a log axis, breakdowns included.

The reweighted(rho=0.5)-minus-cocolasso delta is annotated above that pair
(the paper's actual default vs. the baseline) so the main comparison doesn't
require reading two bar heights and subtracting mentally.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = "results/summary_median.csv"
OUT_PATH = "results/cocolasso_vs_reweighted.png"
FAIL_THRESHOLD = 1000  # median PE above this is a breakdown, not a real estimate

COLOR_COCO = "#6b6a66"
COLOR_REW05 = "#2a78d6"
COLOR_REW0 = "#eb6834"
COLOR_BETTER = "#0ca30c"
COLOR_WORSE = "#d03b3b"

SIGMAS = [0.75, 1.0, 1.25]
NORMS = ["frobenius", "max"]


def get_pe(df, algo, norm, sigma, rho=None):
    mask = (df["algorithm"] == algo) & (df["norm"] == norm) & (df["sigma_a"] == sigma)
    if rho is not None:
        mask &= df["rho"] == rho
    return float(df[mask]["median_pe"].iloc[0])


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["lam_requested"] == "1.0"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150, sharey=True)

    x = np.arange(len(SIGMAS))
    width = 0.27

    for ax, norm in zip(axes, NORMS):
        coco_vals = [get_pe(df, "cocolasso", norm, s) for s in SIGMAS]
        rew05_vals = [get_pe(df, "reweighted", norm, s, rho=0.5) for s in SIGMAS]
        rew0_vals = [get_pe(df, "reweighted", norm, s, rho=0.0) for s in SIGMAS]

        # cap the axis to the real (non-failed) values so failed bars don't
        # need a log scale; failed configurations get a text label instead.
        real_vals = coco_vals + rew05_vals + [v for v in rew0_vals if v <= FAIL_THRESHOLD]
        ymax = max(real_vals) * 1.35

        rew0_plot = [v if v <= FAIL_THRESHOLD else 0 for v in rew0_vals]

        b1 = ax.bar(x - width, coco_vals, width, label="cocolasso", color=COLOR_COCO)
        b2 = ax.bar(x, rew05_vals, width, label="reweighted (ρ=0.5)", color=COLOR_REW05)
        b3 = ax.bar(x + width, rew0_plot, width, label="reweighted (ρ=0)", color=COLOR_REW0)

        for xi, cv, rv in zip(x, coco_vals, rew05_vals):
            top = max(cv, rv)
            delta = rv - cv
            color = COLOR_BETTER if delta < 0 else COLOR_WORSE
            sign = "−" if delta < 0 else "+"
            ax.annotate(f"{sign}{abs(delta):.2f}", (xi, top), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=color, fontweight="bold")

        for bars, vals in ((b1, coco_vals), (b2, rew05_vals)):
            for bar, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8.5, color="white",
                            fontweight="medium")

        for bar, v_real in zip(b3, rew0_vals):
            cx = bar.get_x() + bar.get_width() / 2
            if v_real <= FAIL_THRESHOLD:
                ax.annotate(f"{v_real:.2f}", (cx, 0), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            color="white", fontweight="medium")
            else:
                ax.annotate("failed", (cx, 0), xytext=(0, 6), textcoords="offset points",
                             ha="center", va="bottom", rotation=90, fontsize=9,
                             color=COLOR_WORSE, fontweight="bold")

        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\sigma_a$={s}" for s in SIGMAS], fontsize=10)
        ax.set_title(f"{norm} norm", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("median PE", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("cocolasso vs. reweighted (ρ=0.5, ρ=0), by projection norm and noise level "
                 "($\\lambda$=1, n=100, p=250, 20 reps)\n"
                 "label above the first two bars = reweighted(ρ=0.5) − cocolasso "
                 "(green = reweighted wins, red = cocolasso wins); \"failed\" = median PE "
                 f"> {FAIL_THRESHOLD:,.0f} (numerical breakdown, not a real estimate)",
                 fontsize=10, y=1.12)
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
