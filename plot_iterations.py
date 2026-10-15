"""Exploratory plots of the per-iteration diagnostics from track_iterations.py.

    python plot_iterations.py                         # every setting found
    python plot_iterations.py --setting ar_sigma1.0_datta_zhang

Per setting, in results/iteration_tracking/<setting>/:

  iterations.png        each metric against iteration t
  iterations_delta.png  the same, as the per-seed change from t = 0 as a
                        fraction: norms divided by their t = 0 value, the
                        top-s overlap by s, overlap_prev (no delta) by k

and across settings, in results/iteration_tracking/:

  overview_<metric>.png  one small panel per setting for a few headline metrics

Encoding: colour = projection norm, line style = algorithm. The reweighted path
is a solid line (median over seeds, IQR band); CoCoLasso has a single iteration
and is drawn as a dashed horizontal line at its median in the colour of its
norm. The overlap metrics are integer counts, so they show the mean instead.

Seeds converge at different t. A converged path is a fixed point -- every
further iteration rebuilds the same Sigma_t and beta_t -- so each trajectory is
extended with its last value up to the longest one, and overlap_prev with
|top-k(beta_last)|, which is what comparing beta_last with itself gives. The
x-axis stops at the 95th percentile of iteration counts.

The delta figure matters because the pairing is exact: within a seed every
iteration sees the same data, so metric_t - metric_0 removes the between-seed
spread that otherwise swamps changes along the path. Dividing by metric_0 puts
blocks of very different size (S x S vs Sc x Sc) on a common scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "results" / "iteration_tracking"
ARMS = ["reweighted_max", "reweighted_frobenius", "cocolasso_max", "cocolasso_frobenius"]
COLORS = {"frobenius": "#2a78d6", "max": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COUNT_METRICS = {"overlap_prev", "top3_overlap"}

PANELS = [   # (key, title) -- 4 x 4 grid, row-major
    ("cov_full_max", r"$\|\Sigma_t-\Sigma_X\|_{\max}$ full"),
    ("cov_SS_max", r"$\|\cdot\|_{\max}$  $S\times S$"),
    ("cov_SSc_max", r"$\|\cdot\|_{\max}$  $S\times S^c$"),
    ("cov_ScSc_max", r"$\|\cdot\|_{\max}$  $S^c\times S^c$"),
    ("cov_full_fro", r"$\|\Sigma_t-\Sigma_X\|_F$ full"),
    ("cov_SS_fro", r"$\|\cdot\|_F$  $S\times S$"),
    ("cov_SSc_fro", r"$\|\cdot\|_F$  $S\times S^c$"),
    ("cov_ScSc_fro", r"$\|\cdot\|_F$  $S^c\times S^c$"),
    ("beta_full_l2", r"$\|\hat\beta_t-\beta^*\|_2$ full"),
    ("beta_S_l2", r"$\|\cdot\|_2$ on $S$"),
    ("beta_Sc_l2", r"$\|\cdot\|_2$ on $S^c$"),
    ("overlap_prev", r"$|$top-$k(\hat\beta_t)\cap$top-$k(\hat\beta_{t-1})|$, mean"),
    ("beta_full_l1", r"$\|\hat\beta_t-\beta^*\|_1$ full"),
    ("beta_S_l1", r"$\|\cdot\|_1$ on $S$"),
    ("beta_Sc_l1", r"$\|\cdot\|_1$ on $S^c$"),
    ("top3_overlap", r"$|$top-3$(\hat\beta_t)\cap$top-3$(\beta^*)|$, mean"),
]
OVERVIEW = ["cov_full_max", "cov_SS_max", "beta_full_l2", "top3_overlap", "overlap_prev"]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_arm(setting_dir: Path, arm: str):
    """Merge the arm's file (or its seed-chunk parts) into one dict, or None."""
    files = ([setting_dir / f"{arm}.json"] if (setting_dir / f"{arm}.json").exists()
             else sorted((setting_dir / "parts").glob(f"{arm}.seed*.json")))
    if not files:
        return None
    docs = [json.loads(f.read_text()) for f in files]
    reps = sorted((r for d in docs for r in d["reps"]), key=lambda r: r["seed"])
    return {"config": docs[0]["config"], "support": docs[0]["support"], "reps": reps}


def trajectories(run, key: str, k: int) -> np.ndarray:
    """seeds x T_max array of `key`, each row extended with its last value."""
    T = max(len(r["rows"]) for r in run["reps"])
    out = np.full((len(run["reps"]), T), np.nan)
    for i, rep in enumerate(run["reps"]):
        vals = [np.nan if row[key] is None else row[key] for row in rep["rows"]]
        if key == "overlap_prev":
            fill = min(k, rep["rows"][-1]["nnz"])
        else:
            fill = vals[-1]
        out[i, :len(vals)] = vals
        out[i, len(vals):] = fill
    return out


def summarise(A: np.ndarray, key: str):
    """(centre, lo, hi) per column: mean for counts, median + IQR otherwise."""
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        if key in COUNT_METRICS:
            m = np.nanmean(A, axis=0)
            return m, None, None
        return (np.nanmedian(A, axis=0), np.nanpercentile(A, 25, axis=0),
                np.nanpercentile(A, 75, axis=0))


def change(A: np.ndarray, ref: np.ndarray, key: str, k: int, s: int) -> np.ndarray:
    """Delta-figure value as a fraction, per seed: (m_t - m_0) / m_0 for
    norms, (m_t - m_0) / s for the top-s overlap, and overlap_prev / k (it has
    no t = 0 value to subtract)."""
    if key == "overlap_prev":
        return A / k
    if key == "top3_overlap":
        return (A - ref) / s
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ref != 0, (A - ref) / ref, np.nan)


BLOCK_TEX = {"full": "full", "SS": r"$S\times S$", "SSc": r"$S\times S^c$",
             "ScSc": r"$S^c\times S^c$", "S": r"on $S$", "Sc": r"on $S^c$"}
NORM_TEX = {"max": r"\max", "fro": "F", "l2": "2", "l1": "1"}


def delta_title(key: str, k: int, s: int) -> str:
    """Panel title for the delta figure, naming the normalisation factor."""
    if key == "overlap_prev":
        return rf"$|$top-$k(\hat\beta_t)\cap$top-$k(\hat\beta_{{t-1}})|\ /\ {k}$, mean"
    if key == "top3_overlap":
        return rf"$\Delta|$top-{s}$(\hat\beta_t)\cap$top-{s}$(\beta^*)|\ /\ {s}$, mean"
    kind, block, norm = key.split("_")
    err = r"\Sigma_{}-\Sigma_X" if kind == "cov" else r"\hat\beta_{}-\beta^*"
    nt = NORM_TEX[norm]
    return (rf"$\Delta\|{err.format('t')}\|_{{{nt}}}\ /\ \|{err.format('0')}\|_{{{nt}}}$"
            f"  {BLOCK_TEX[block]}")


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def draw(ax, runs, key, k, t_max, delta):
    """Reweighted curves and CoCoLasso reference lines for one metric."""
    s = next(iter(runs.values()))["config"]["true_k"]
    base = {}
    for arm in ("reweighted_max", "reweighted_frobenius"):
        if arm in runs:
            base[arm.split("_", 1)[1]] = trajectories(runs[arm], key, k)
    for arm in ("reweighted_max", "reweighted_frobenius"):
        run = runs.get(arm)
        if run is None:
            continue
        norm = arm.split("_", 1)[1]
        A = base[norm][:, :t_max + 1]
        if delta:
            A = change(A, A[:, :1], key, k, s)
        m, lo, hi = summarise(A, key)
        t = np.arange(A.shape[1])
        if lo is not None:
            ax.fill_between(t, lo, hi, color=COLORS[norm], alpha=0.15, lw=0)
        ax.plot(t, m, "-", color=COLORS[norm], lw=2, marker="o", ms=3.5,
                markeredgecolor="white", markeredgewidth=0.6)
    for arm in ("cocolasso_max", "cocolasso_frobenius"):
        run = runs.get(arm)
        norm = arm.split("_", 1)[1]
        if run is None or key == "overlap_prev":
            continue
        A = trajectories(run, key, k)[:, :1]
        if delta:
            # Paired by seed against the reweighted path's own t = 0, which is
            # CoCoLasso up to projection tolerance: this line should sit at 0.
            if norm not in base:
                continue
            ref = {r["seed"]: i for i, r in enumerate(runs[f"reweighted_{norm}"]["reps"])}
            rows = [ref.get(r["seed"]) for r in run["reps"]]
            keep = [i for i, j in enumerate(rows) if j is not None]
            A = change(A[keep], base[norm][[rows[i] for i in keep], :1], key, k, s)
        m, _, _ = summarise(A, key)
        ax.axhline(m[0], color=COLORS[norm], ls=(0, (4, 3)), lw=1.4)


def legend_handles(runs):
    hs, labels = [], []
    for arm in ARMS:
        if arm not in runs:
            continue
        algo, norm = arm.split("_", 1)
        n = len(runs[arm]["reps"])
        if algo == "reweighted":
            it = np.array([r["iters"] for r in runs[arm]["reps"]])
            h = plt.Line2D([], [], color=COLORS[norm], lw=2, marker="o", ms=4)
            lab = f"reweighted · {norm}  (n={n}, iters median {np.median(it):.0f}, max {it.max()})"
        else:
            h = plt.Line2D([], [], color=COLORS[norm], lw=1.4, ls=(0, (4, 3)))
            lab = f"CoCoLasso · {norm}  (n={n})"
        hs.append(h)
        labels.append(lab)
    return hs, labels


def setting_title(cfg, support):
    beta = cfg["beta_preset"] + (f" α={cfg['alpha']}" if cfg.get("alpha") else "")
    return (f"{cfg['model'].upper()}, n={cfg['n']}, p={cfg['p']}, σ_a={cfg['sigma_a']}, "
            f"σ_e={cfg['sigma_e']}, β*={beta}, λ={cfg['lam']}, k={cfg['k']}, "
            f"ρ={cfg['rho']}   |   S = top-{cfg['true_k']} of |β*| = {support}")


def t_limit(runs):
    its = [r["iters"] for a in ("reweighted_max", "reweighted_frobenius") if a in runs
           for r in runs[a]["reps"]]
    return int(np.ceil(np.percentile(its, 95))) - 1 if its else 0


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plot_setting(setting_dir: Path, runs: dict, delta: bool):
    any_run = next(iter(runs.values()))
    k = any_run["config"]["k"]
    t_max = t_limit(runs)
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), dpi=130)
    s = any_run["config"]["true_k"]
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        draw(ax, runs, key, k, t_max, delta)
        if delta:
            title = delta_title(key, k, s)
            if key != "overlap_prev":           # a level, not a change from t = 0
                ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_xlim(-0.3, t_max + 0.3)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("iteration t  (t = 0 is CoCoLasso)", fontsize=9, color=MUTED)
    hs, labels = legend_handles(runs)
    fig.legend(hs, labels, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.962))
    what = ("Per-seed change from t = 0, Δ = m_t − m_0, divided by the factor in each "
            "panel title (median, IQR)" if delta
            else "Metrics along the homotopy (median over seeds, IQR band)")
    fig.suptitle(f"{what}\n{setting_title(any_run['config'], any_run['support'])}",
                 fontsize=11, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = setting_dir / ("iterations_delta.png" if delta else "iterations.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_overview(all_runs: dict, key: str, delta: bool = False):
    title = dict(PANELS)[key]
    names = sorted(all_runs)
    rows = sorted({n.split("_", 2)[0] + "_" + n.split("_", 2)[1] for n in names},
                  key=lambda s: (s.split("_")[0], float(s.split("sigma")[1])))
    cols = sorted({n.split("_", 2)[2] for n in names},
                  key=lambda s: (s != "datta_zhang", s))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.6 * len(cols), 2.6 * len(rows)),
                             dpi=130, squeeze=False, sharex=False)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            runs = all_runs.get(f"{r}_{c}")
            style(ax)
            if not runs:
                ax.set_visible(False)
                continue
            k = next(iter(runs.values()))["config"]["k"]
            t_max = t_limit(runs)
            draw(ax, runs, key, k, t_max, delta)
            ax.set_xlim(-0.3, t_max + 0.3)
            if i == 0:
                ax.set_title(c.replace("weak_sparse_a", "weak_sparse α="), fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(r.replace("_sigma", "  σ_a="), fontsize=10, color=INK)
            if i == len(rows) - 1:
                ax.set_xlabel("iteration t", fontsize=9, color=MUTED)
    any_runs = next(iter(all_runs.values()))
    hs, labels = legend_handles(any_runs)
    labels = [lab.split("  (")[0] for lab in labels]
    fig.legend(hs, labels, loc="upper center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"{title} along the homotopy, every setting (median over seeds, IQR band; "
                 f"y-axes not shared)", fontsize=12, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = ROOT / f"overview_{key}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--setting", default=None, help="only this setting")
    args = ap.parse_args()
    root = Path(args.root)

    dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "smoke")
    if args.setting:
        dirs = [root / args.setting]
    all_runs = {}
    for d in dirs:
        runs = {a: r for a in ARMS if (r := load_arm(d, a)) is not None}
        if not runs:
            continue
        all_runs[d.name] = runs
        counts = ", ".join(f"{a}={len(r['reps'])}" for a, r in runs.items())
        for delta in (False, True):
            out = plot_setting(d, runs, delta)
        print(f"{d.name}: {counts}  -> {out.parent}")
    if len(all_runs) > 1:
        for key in OVERVIEW:
            print(f"saved {plot_overview(all_runs, key)}")


if __name__ == "__main__":
    main()
