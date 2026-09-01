"""Flatten the sweep's result JSONs into one CSV, one row per repetition.

    python export_runs_csv.py                          # results/sweep_p500 -> results/sweep_p500_runs.csv
    python export_runs_csv.py --root results/sweep_p500 --out runs.csv
    python export_runs_csv.py --summary                 # one row per run instead

Each row is a single *repetition* -- one fitted estimate on one dataset -- not
an aggregate over repetitions. `results/summary_median.csv` is the aggregated
view; this is the raw one underneath it, so medians, quantiles, paired
differences and breakdown counts can all be recomputed downstream without
re-running anything.

Every row carries the full configuration that produced it, including the
algorithm parameters the run actually used (`effective_algo_params` records the
defaults an algorithm never saw, so `algo_psd_floor` is populated even when
nothing was passed on the command line).

Chunked arms merge for free. `reweighted_max` is submitted as four 25-rep jobs
at seeds 0/25/50/75 to survive preemption; since every row carries its own
`seed`, expanding the four part files reconstructs the full 100-repetition set
with no merge step. When a merged top-level file exists, its parts are skipped
so the rows are not counted twice.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# config fields copied verbatim onto every row
CONFIG_SCALARS = ["model", "n", "p", "sigma_a", "sigma_e", "beta_preset", "alpha",
                  "lam_requested", "algorithm", "norm", "k", "true_k", "snr"]

# algo_params worth a column; the union across algorithms, blank where an
# algorithm does not take one (cocolasso has no rho, naive has no gamma rule).
ALGO_PARAMS = ["rho", "gamma_rule", "gamma1", "growth", "f_tol", "min_iter",
               "max_iter", "weight_clip", "admm_rho", "admm_max_iter",
               "psd_floor", "ridge", "refit_thresh"]

PER_REP = ["mse", "pe", "topk", "nnz", "time", "lam"]

COLUMNS = (["setting", "arm", "rep_index", "seed"]
           + CONFIG_SCALARS
           + PER_REP
           + ["algo_" + p for p in ALGO_PARAMS]
           + ["n_reps_in_file", "homotopy_iters_last", "runtime_sec_file",
              "source_file"])

SUMMARY_METRICS = ["mse_mean", "mse_std", "mse_se", "pe_mean", "pe_std", "pe_se",
                   "topk_rate", "nnz_mean", "time_mean", "time_std", "time_total",
                   "lam_mean", "lam_std"]

SUMMARY_COLUMNS = (["setting", "arm"] + CONFIG_SCALARS + SUMMARY_METRICS
                   + ["algo_" + p for p in ALGO_PARAMS]
                   + ["n_reps_in_file", "homotopy_iters_last", "runtime_sec_file",
                      "source_file"])


def arm_of(path: Path) -> tuple[str, int | None]:
    """(arm name, chunk seed) from a filename.

    'reweighted_max.json'         -> ('reweighted_max', None)
    'reweighted_max.seed25.json'  -> ('reweighted_max', 25)
    """
    stem = path.name[: -len(".json")]
    if ".seed" in stem:
        arm, _, seed = stem.partition(".seed")
        return arm, int(seed)
    return stem, None


def json_files(root: Path) -> list[Path]:
    """Every result file under root, with superseded chunk parts dropped.

    A part is superseded once its merged top-level file exists; including both
    would double-count those repetitions.
    """
    out = []
    for f in sorted(root.rglob("*.json")):
        if f.parent.name == "parts":
            arm, _ = arm_of(f)
            if (f.parent.parent / f"{arm}.json").exists():
                continue
        out.append(f)
    return out


def rows_from(path: Path, root: Path, summary: bool):
    """Yield one dict per repetition (or per run, if summary)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        # A job killed mid-write, or still running: main.py writes its JSON
        # only after the final repetition, so a partial file is expected while
        # a sweep is in flight. Skip it rather than aborting the export.
        print(f"  skipping unreadable {path.name}: {e}", file=sys.stderr)
        return

    setting = path.parent.name if path.parent.name != "parts" else path.parent.parent.name
    arm, _chunk = arm_of(path)
    rel = str(path.relative_to(root))

    for run in doc.get("runs", []):
        cfg = run.get("config", {})
        algo = cfg.get("algo_params", {}) or {}

        base = {"setting": setting, "arm": arm, "source_file": rel,
                "n_reps_in_file": cfg.get("n_reps"),
                "homotopy_iters_last": run.get("homotopy_iters"),
                "runtime_sec_file": run.get("runtime_sec")}
        for key in CONFIG_SCALARS:
            base[key] = cfg.get(key)
        for key in ALGO_PARAMS:
            base["algo_" + key] = algo.get(key)

        if summary:
            metrics = run.get("metrics", {})
            yield {**base, **{m: metrics.get(m) for m in SUMMARY_METRICS}}
            continue

        per_rep = run.get("per_rep", {})
        seeds = cfg.get("seeds") or []
        n = len(per_rep.get("mse", []))
        for i in range(n):
            row = dict(base)
            row["rep_index"] = i
            # The seed is the identity of the dataset: rows sharing a seed
            # within a setting were fit on bit-identical data, which is what
            # makes the arms comparable pairwise.
            row["seed"] = seeds[i] if i < len(seeds) else None
            for key in PER_REP:
                vals = per_rep.get(key) or []
                row[key] = vals[i] if i < len(vals) else None
            yield row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/sweep_p500",
                    help="directory tree of result JSONs [default: %(default)s]")
    ap.add_argument("--out", default=None,
                    help="output CSV [default: <root>_runs.csv, or _summary.csv]")
    ap.add_argument("--summary", action="store_true",
                    help="one row per run with the aggregate metrics, instead "
                         "of one row per repetition")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else Path(
        f"{root}_{'summary' if args.summary else 'runs'}.csv")
    columns = SUMMARY_COLUMNS if args.summary else COLUMNS

    files = json_files(root)
    rows = [r for f in files for r in rows_from(f, root, args.summary)]
    if not rows:
        print(f"no rows found under {root}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    settings = {r["setting"] for r in rows}
    arms = {r["arm"] for r in rows}
    print(f"wrote {len(rows)} rows to {out}")
    print(f"  {len(files)} files, {len(settings)} settings, {len(arms)} arms")
    if not args.summary:
        complete = sum(1 for r in rows if r.get("pe") is not None)
        print(f"  {complete} rows with a PE value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
