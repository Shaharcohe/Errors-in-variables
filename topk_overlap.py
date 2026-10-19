"""Top-10 support overlaps of the final beta for the p=500 sweep.

    python topk_overlap.py

No refitting: track_iterations.py stores every beta_t of every sweep fit
(bit-identical to sweep_p500, float32), so the final beta of each seed is the
last row of its block in results/iteration_tracking/<setting>/<arm>.npz
(reweighted_max is merged from its 4 seed chunks under parts/).

Per repetition:
  top10_vs_top10  |top-10(beta_hat) & top-10(beta*)|, 0..10
  top10_vs_top3   |top-10(beta_hat) & top-3(beta*)|,  0..3
top-m ranks |beta_j| among the nonzeros only (as in track_iterations.top_k), so
an estimate with nnz < 10 contributes fewer than 10 indices. datta_zhang has
only 3 nonzeros, so its "true top-10" is the support and the two metrics agree.

Arms without tracked betas (twostep_*) are taken from the sweep CSV instead,
where test.compute_metrics recorded the same two overlaps at fit time.

Writes results/topk_overlap_runs.csv (one row per repetition) and
results/topk_overlap_summary.csv (mean and se per setting x arm).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import config
from track_iterations import P, top_k

ROOT = Path("results/iteration_tracking")
SWEEP_CSV = "results/sweep_p500_runs.csv"
ARMS = ["cocolasso_max", "cocolasso_frobenius", "reweighted_max", "reweighted_frobenius"]
CHUNKS = [0, 25, 50, 75]


def final_betas(setting_dir: Path, arm: str):
    """(seeds, final betas) for one arm, merging seed chunks when needed."""
    files = ([setting_dir / f"{arm}.npz"] if (setting_dir / f"{arm}.npz").exists()
             else [setting_dir / "parts" / f"{arm}.seed{s}.npz" for s in CHUNKS])
    seeds, betas = [], []
    for f in files:
        z = np.load(f)
        ends = np.cumsum(z["iters"]) - 1
        seeds += z["seeds"].tolist()
        betas.append(z["beta"][ends].astype(float))
    return seeds, np.concatenate(betas)


def parse_setting(name: str):
    model, sig, beta = name.split("_", 2)
    sigma_a = float(sig.removeprefix("sigma"))
    if beta == "datta_zhang":
        return model, sigma_a, beta, config.make_beta("datta_zhang", P)
    alpha = float(beta.rsplit("_a", 1)[1])
    return model, sigma_a, beta, config.make_beta("weak_sparse", P, alpha=alpha)


def main():
    # MSE from the sweep, to check the stored betas are the sweep's fits.
    sweep_mse = {(r["setting"], r["arm"], int(r["seed"])): float(r["mse"])
                 for r in csv.DictReader(open(SWEEP_CSV))}

    rows, worst = [], 0.0
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "smoke"):
        model, sigma_a, beta_tag, beta_star = parse_setting(d.name)
        true10 = set(top_k(beta_star, 10).tolist())
        true3 = set(top_k(beta_star, 3).tolist())
        for arm in ARMS:
            seeds, B = final_betas(d, arm)
            for seed, b in zip(seeds, B):
                top10 = set(top_k(b, 10).tolist())
                mse = float(np.sum((b - beta_star) ** 2))
                ref = sweep_mse.get((d.name, arm, seed))
                if ref is not None:
                    worst = max(worst, abs(mse - ref) / max(ref, 1e-12))
                rows.append({"setting": d.name, "model": model, "sigma_a": sigma_a,
                             "beta": beta_tag, "arm": arm, "seed": seed,
                             "top10_vs_top10": len(top10 & true10),
                             "top10_vs_top3": len(top10 & true3),
                             "nnz": int(np.sum(np.abs(b) > 1e-10)), "mse": mse})
    print(f"{len(rows)} fits; max rel. MSE diff vs sweep CSV: {worst:.1e}")

    # Arms with no tracked betas: overlaps recorded by the sweep itself.
    extra = [r for r in csv.DictReader(open(SWEEP_CSV))
             if r["arm"] not in ARMS and r.get("top10_vs_top10")]
    for r in extra:
        model, sigma_a, beta_tag, _ = parse_setting(r["setting"])
        rows.append({"setting": r["setting"], "model": model, "sigma_a": sigma_a,
                     "beta": beta_tag, "arm": r["arm"], "seed": int(r["seed"]),
                     "top10_vs_top10": int(float(r["top10_vs_top10"])),
                     "top10_vs_top3": int(float(r["top10_vs_top3"])),
                     "nnz": int(r["nnz"]), "mse": float(r["mse"])})
    arms = ARMS + sorted({r["arm"] for r in extra})
    print(f"{len(extra)} more fits from the sweep CSV: {sorted({r['arm'] for r in extra})}")

    with open("results/topk_overlap_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for setting in sorted({r["setting"] for r in rows}):
        for arm in arms:
            sub = [r for r in rows if r["setting"] == setting and r["arm"] == arm]
            if not sub:
                continue
            out = {"setting": setting, "arm": arm, "n": len(sub)}
            for m in ("top10_vs_top10", "top10_vs_top3"):
                x = np.array([r[m] for r in sub], dtype=float)
                out[f"{m}_mean"] = round(x.mean(), 3)
                out[f"{m}_se"] = round(x.std(ddof=1) / np.sqrt(len(x)), 3)
            summary.append(out)
    with open("results/topk_overlap_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    def short(a):
        return (a.replace("cocolasso", "coco").replace("reweighted", "rw")
                 .replace("twostep", "2s").replace("frobenius", "fro"))
    print(f"\n{'setting':30s}" + "".join(f"{short(a):>16s}" for a in arms))
    print(f"{'(mean top10/10 | top10/3)':30s}")
    cell = {(s["setting"], s["arm"]): s for s in summary}
    for setting in sorted({s["setting"] for s in summary}):
        print(f"{setting:30s}" + "".join(
            f"{c['top10_vs_top10_mean']:>9.2f} |{c['top10_vs_top3_mean']:>5.2f}"
            if (c := cell.get((setting, a))) else f"{'-':>16s}" for a in arms))


if __name__ == "__main__":
    main()
