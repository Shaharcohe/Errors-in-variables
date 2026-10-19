"""LaTeX tables in the style of Datta & Zhang (2017), Table 1, for the p=500 sweep.

    python make_latex_table.py                      # all beta* presets
    python make_latex_table.py --beta datta_zhang,weak_sparse_a1.5

One table per beta* preset, each written to results/table_<beta>.tex, plus all
of them together in results/tables_p500.tex.

Columns per tau: CoCoLasso (max), ours (max), ours (Frobenius). Rows: median
PE / MSE with bootstrap se of the median, the paired win rate of each of our
arms against CoCoLasso (max) on the same seeds, and median runtime.
"""
from __future__ import annotations

import argparse
import collections
import csv

import numpy as np

CSV_IN = "results/sweep_p500_runs.csv"
BASE = "cocolasso_max"
ARMS = [BASE, "reweighted_max", "reweighted_frobenius"]
HEAD = ["CoCo", r"RW$_{\max}$", r"RW$_F$"]
TAUS = ["0.75", "1.0", "1.25"]
MODELS = ["ar", "cs"]

# setting suffix -> description of beta* for the caption
BETAS = {
    "datta_zhang": r"$\beta^* = (3, 1.5, 0, 0, 2, 0, \dots, 0)^\top$",
    "weak_sparse_a1.0": r"weakly sparse $\beta^*_j = \pm 3 j^{-\eta}$, $\eta = 1.0$",
    "weak_sparse_a1.5": r"weakly sparse $\beta^*_j = \pm 3 j^{-\eta}$, $\eta = 1.5$",
    "weak_sparse_a2.0": r"weakly sparse $\beta^*_j = \pm 3 j^{-\eta}$, $\eta = 2.0$",
}


def boot_se_median(x: np.ndarray, B: int, rng: np.random.Generator) -> float:
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return float(np.median(x[idx], axis=1).std(ddof=1))


def win_cell(w: float) -> str:
    """Win % shaded green above 50, red below; darker the further from 50."""
    shade = round(5 + 35 * abs(w - 50) / 50)
    color = "green" if w > 50 else "red" if w < 50 else "gray"
    return rf"\cellcolor{{{color}!{shade}}}{w:.0f}"


def make_table(runs, beta: str, B: int, rng: np.random.Generator) -> list[str]:
    ncol = len(ARMS)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Summary statistics for the additive error simulation study "
        rf"with {BETAS[beta]} ($n=100$, $p=500$, $\lambda=1$ fixed) based on 100 "
        r"replications. Reported numbers are medians; standard errors (se) are "
        r"computed by bootstrap. ``Win'' is the percentage of paired replications "
        r"in which RW (reweighted CoCoLasso) has lower error than CoCoLasso (max "
        r"norm), shaded green above 50\% and red below. Bold marks the lowest "
        r"median within each $\tau$. Time is the median wall-clock time per fit "
        r"in seconds.}",
        rf"\label{{tab:{beta.replace('_', '-')}}}",
        r"\setlength{\tabcolsep}{4pt}",
        # shrink to \textwidth only if wider; never scale up
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        r"\begin{tabular}{c|c|" + "|".join(["c" * ncol] * len(TAUS)) + "}",
        r"\hline",
        " & & " + " & ".join(
            rf"\multicolumn{{{ncol}}}{{c{'|' if i < len(TAUS) - 1 else ''}}}{{$\tau = {t}$}}"
            for i, t in enumerate(TAUS)) + r" \\",
        " & & " + " & ".join(HEAD * len(TAUS)) + r" \\",
        r"\hline",
    ]

    for model in MODELS:
        cells = collections.defaultdict(list)  # row label -> cells
        for tau in TAUS:
            setting = f"{model}_sigma{tau}_{beta}"
            base = runs[(setting, BASE)]
            if not base:
                raise SystemExit(f"no rows for {setting} / {BASE}")
            med = collections.defaultdict(list)  # name -> median per arm
            for arm in ARMS:
                arm_runs = runs[(setting, arm)]
                seeds = sorted(set(base) & set(arm_runs), key=int)
                for m, name in (("pe", "PE"), ("mse", "MSE")):
                    x = np.array([float(arm_runs[s][m]) for s in seeds])
                    med[name].append(np.median(x))
                    cells[f"se({name})"].append(f"{boot_se_median(x, B, rng):.2f}")
                    if arm == BASE:
                        cells[f"Win {name} (\\%)"].append("--")
                    else:
                        b = np.array([float(base[s][m]) for s in seeds])
                        cells[f"Win {name} (\\%)"].append(win_cell(100 * np.mean(x < b)))
                med["Time (s)"].append(np.median([float(arm_runs[s]["time"]) for s in seeds]))
            # best (lowest) median per tau block in bold
            for name, fmt in (("PE", ".2f"), ("MSE", ".2f"), ("Time (s)", ".1f")):
                best = int(np.argmin(med[name]))
                cells[name] += [rf"\textbf{{{v:{fmt}}}}" if i == best else f"{v:{fmt}}"
                                for i, v in enumerate(med[name])]
        order = ["PE", "se(PE)", "MSE", "se(MSE)", "Win PE (\\%)", "Win MSE (\\%)",
                 "Time (s)"]
        lines.append(rf"\multirow{{{len(order)}}}{{*}}{{{model.upper()}}}")
        for name in order:
            lines.append(f" & {name} & " + " & ".join(cells[name]) + r" \\")
        lines.append(r"\hline")

    lines += [r"\end{tabular}}", r"\end{table}"]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_IN)
    ap.add_argument("--beta", default=",".join(BETAS),
                    help=f"comma-separated setting suffixes {list(BETAS)}")
    ap.add_argument("--out", default="results/tables_p500.tex",
                    help="combined file; each table also goes to results/table_<beta>.tex")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap resamples")
    args = ap.parse_args()

    # runs[(setting, arm)][seed] = row
    runs = collections.defaultdict(dict)
    for r in csv.DictReader(open(args.csv)):
        runs[(r["setting"], r["arm"])][r["seed"]] = r

    rng = np.random.default_rng(0)
    tables = []
    for beta in args.beta.split(","):
        text = "\n".join(make_table(runs, beta, args.B, rng)) + "\n"
        path = f"results/table_{beta}.tex"
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {path}")
        tables.append(text)
    with open(args.out, "w") as f:
        f.write("\n".join(tables))
    print(f"wrote {args.out}  (needs \\usepackage{{multirow}}, graphicx and \\usepackage[table]{{xcolor}})")


if __name__ == "__main__":
    main()
