"""Plot median PE by configuration from results/summary_median.csv.

One panel per sigma_a, log-scaled x-axis (values span ~11 orders of
magnitude once naive/rho=0 breakdowns are included), dots rather than bars
since bar length on a log axis misrepresents the value. Reads the CSV
produced by the ad-hoc extraction snippets used during the lambda=1 sweep;
regenerate that CSV before re-running this if new results have landed.
"""
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "results/summary_median.csv"
OUT_PATH = "results/median_pe_sweep.png"
CRIT_THRESHOLD = 1000

ROW_ORDER = [
    ("cocolasso", "frobenius", None),
    ("cocolasso", "max", None),
    ("cocolasso_refit", "frobenius", None),
    ("cocolasso_refit", "max", None),
    ("reweighted", "frobenius", 0.5),
    ("reweighted", "max", 0.5),
    ("reweighted", "frobenius", 0.0),
    ("reweighted", "max", 0.0),
    ("reweighted_refit", "frobenius", 0.5),
    ("reweighted_refit", "max", 0.5),
    ("naive", "frobenius", None),
]
SIGMAS = [0.75, 1.0, 1.25]


def row_label(algo, norm, rho):
    if algo == "naive":
        return "naive"
    if algo == "reweighted":
        return f"reweighted (ρ={rho}) · {norm}"
    return f"{algo} · {norm}"


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["lam_requested"] == "1.0"] if df["lam_requested"].dtype == object else df[df["lam_requested"] == 1.0]
    # rho column: NaN for algorithms where it's not applicable (cocolasso*, naive)
    df["rho"] = df["rho"].astype(object).where(df["rho"].notna(), None)

    fig, axes = plt.subplots(len(SIGMAS), 1, figsize=(9, 11), dpi=150)

    for ax, sigma in zip(axes, SIGMAS):
        sub = df[df["sigma_a"] == sigma]

        labels, values = [], []
        for algo, norm, rho in ROW_ORDER:
            mask = (sub["algorithm"] == algo) & (sub["norm"] == norm)
            mask &= sub["rho"].isna() if rho is None else (sub["rho"] == rho)
            match = sub[mask]
            labels.append(row_label(algo, norm, rho))
            values.append(float(match["median_pe"].iloc[0]) if len(match) else None)

        y = list(range(len(labels)))[::-1]
        present = [(yi, v) for yi, v in zip(y, values) if v is not None]
        winner_val = min(v for _, v in present)

        for yi, v in zip(y, values):
            if v is None:
                continue
            if v == winner_val:
                color, size, zorder = "#2a78d6", 90, 3
            elif v > CRIT_THRESHOLD:
                color, size, zorder = "#d03b3b", 60, 2
            else:
                color, size, zorder = "#6b6a66", 55, 2
            ax.scatter([v], [yi], s=size, color=color, zorder=zorder, edgecolor="white", linewidth=0.6)
            label_str = f"{v:.2e}" if v >= 1000 else f"{v:.2f}"
            ax.annotate(label_str, (v, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=color if v == winner_val or v > CRIT_THRESHOLD else "#333")

        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("median PE (log scale)", fontsize=9)
        ax.set_title(f"$\\sigma_a$ = {sigma}", fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="x", which="major", linestyle="-", linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Median PE by configuration, $\\lambda$=1 sweep (n=100, p=250, 20 reps)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
