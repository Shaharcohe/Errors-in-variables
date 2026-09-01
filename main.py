"""Terminal front end for the errors-in-variables experiments.

    python main.py                          walk through the parameters, defaults in [..]
    python main.py --model ar --n_reps 20   run straight through, non-interactively
    python main.py --interactive --p 800    prompt, but start from p = 800

Sweepable arguments take comma-separated lists and main runs their product:

    python main.py --lam 1.0,2.0 --algorithm cocolasso,reweighted --sigma_a 0.75,1.25

Results go to results/<timestamp>.json, one file per invocation, holding every
run's configuration, aggregate metrics and per-repetition values. The seeds are
recorded, so runs that share seeds are paired by construction and can be
compared that way afterwards even though test.py does not compute differences.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import test as testmod

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# name -> (type, default, sweepable, help)
PARAMS = [
    ("model", str, "ar", True, f"covariance model {sorted(config.MODELS)}"),
    ("n", int, 100, True, "sample size"),
    ("p", int, 250, True, "number of covariates"),
    ("sigma_a", float, 1.0, True, "measurement-error sd on the design (paper's tau)"),
    ("sigma_e", float, 3.0, True, "response noise sd (paper's sigma)"),
    ("beta", str, "datta_zhang", True, f"beta* preset {sorted(config.BETA_PRESETS)}"),
    ("alpha", float, 1.0, True, "decay exponent for the weak_sparse beta "
                                "preset (ignored by other presets)"),
    ("lam", str, "1.0", True, "ell-1 penalty weight, or 'cv' to select it by "
                              "corrected k-fold cross-validation"),
    ("algorithm", str, "cocolasso", True, f"{sorted(testmod.ALGORITHMS)}"),
    ("norm", str, "frobenius", True,
     f"projection norm {list(testmod.PROJECTION_SCHEDULES)}; max_then_frobenius "
     f"uses the max norm on the first homotopy iteration only"),
    ("k", int, 10, True, "sparsity level given to the algorithm (an upper bound)"),
    ("true_k", int, 3, False, "true |supp(beta*)|, used only for scoring"),
    ("n_reps", int, 20, False, "repetitions"),
    ("seed", int, 0, False, "first seed; repetition i uses seed + i"),
    ("rho", float, 0.5, True, "GSM damping, reweighted algorithms only"),
    ("cv_folds", int, 5, False, "folds, when lam='cv'"),
    ("cv_lambdas", int, 20, False, "grid size, when lam='cv'"),
    ("cv_ratio", float, 1e-2, False, "smallest lambda as a fraction of the "
                                     "largest, when lam='cv'"),
]

SWEEPABLE = {name for name, _, _, sweep, _ in PARAMS if sweep}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, typ, default, sweep, helptext in PARAMS:
        extra = " (comma-separated list to sweep)" if sweep else ""
        ap.add_argument(f"--{name}", type=str, default=None,
                        help=f"{helptext} [default: {default}]{extra}")
    ap.add_argument("--interactive", action="store_true",
                    help="prompt for parameters even when arguments are given")
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON path [default: results/<timestamp>.json]")
    ap.add_argument("--quiet", action="store_true", help="suppress the summary table")
    ap.add_argument("--progress", action="store_true",
                    help="print a line after every repetition, not just every run "
                         "(useful for slow configurations, e.g. norm=max)")
    return ap


def _lam_ok(v: str) -> bool:
    """lam is either 'cv' or a non-negative number, and arrives as a string."""
    if str(v).lower() == "cv":
        return True
    try:
        return float(v) >= 0
    except ValueError:
        return False


class InputError(Exception):
    """A bad parameter value. `fields` are the parameters to re-ask for."""

    def __init__(self, message: str, fields=()):
        super().__init__(message)
        self.fields = tuple(fields)


# Per-value checks, applied the moment a value is entered so the prompt can be
# repeated rather than failing after all fourteen questions have been asked.
VALIDATORS = {
    "model": lambda v: None if v in config.MODELS
        else f"{v!r} is not one of {sorted(config.MODELS)}",
    "beta": lambda v: None if v in config.BETA_PRESETS
        else f"{v!r} is not one of {sorted(config.BETA_PRESETS)}",
    "algorithm": lambda v: None if v in testmod.ALGORITHMS
        else f"{v!r} is not one of {sorted(testmod.ALGORITHMS)}",
    "norm": lambda v: None if v in testmod.PROJECTION_SCHEDULES
        else f"{v!r} is not one of {list(testmod.PROJECTION_SCHEDULES)}",
    "n": lambda v: None if v >= 1 else "sample size must be at least 1",
    "p": lambda v: None if v >= 1 else "must have at least one covariate",
    "sigma_a": lambda v: None if v >= 0 else "a standard deviation cannot be negative",
    "sigma_e": lambda v: None if v >= 0 else "a standard deviation cannot be negative",
    "lam": lambda v: None if _lam_ok(v) else
        "must be a non-negative number, or 'cv'",
    "cv_folds": lambda v: None if v >= 2 else "need at least 2 folds",
    "cv_lambdas": lambda v: None if v >= 2 else "need at least 2 grid points",
    "cv_ratio": lambda v: None if 0 < v < 1 else "must lie strictly in (0, 1)",
    "k": lambda v: None if v >= 1 else "must select at least one coordinate",
    "true_k": lambda v: None if v >= 1 else "must be at least 1",
    "alpha": lambda v: None if v > 0 else "must be positive",
    "n_reps": lambda v: None if v >= 1 else "need at least one repetition",
    "rho": lambda v: None if 0.0 <= v <= 1.0 else "must lie in [0, 1]",
}


def _clean(s: str) -> str:
    """Whitespace and a byte-order mark, which str.strip() does not remove."""
    return s.strip().strip("﻿").strip()


def parse_values(raw: str, typ, name: str) -> list:
    """A comma-separated string into a validated list of `typ`.

    Raises InputError rather than exiting, so an interactive caller can ask
    again instead of the program dying on a typo.
    """
    out = []
    for piece in str(raw).split(","):
        piece = _clean(piece)
        if not piece:
            continue
        try:
            value = typ(piece)
        except ValueError:
            raise InputError(f"cannot read {piece!r} as {typ.__name__}", [name])
        problem = VALIDATORS.get(name, lambda _v: None)(value)
        if problem:
            raise InputError(f"{problem}", [name])
        out.append(value)
    if not out:
        raise InputError("no value given", [name])
    return out


def ask(name, typ, default, sweep) -> list:
    """Prompt until the answer parses and validates."""
    hint = " (comma-separated to sweep)" if sweep else ""
    while True:
        try:
            raw = _clean(input(f"  {name}{hint} [{default}]: ")) or str(default)
        except EOFError:
            raw = str(default)
        try:
            return parse_values(raw, str if typ is str else typ, name)
        except InputError as e:
            print(f"    -> {name}: {e}. Try again.", file=sys.stderr)


def collect(args, interactive: bool) -> dict:
    """Resolve every parameter to a list of values, prompting if asked to."""
    if interactive:
        print("Parameters -- press enter to accept the default.\n")

    values = {}
    for name, typ, default, sweep, _ in PARAMS:
        raw = getattr(args, name)
        if raw is None and interactive:
            parsed = ask(name, typ, default, sweep)
        else:
            try:
                parsed = parse_values(str(default) if raw is None else raw,
                                      str if typ is str else typ, name)
            except InputError as e:
                raise SystemExit(f"--{name}: {e}")
        values[name] = parsed if sweep else parsed[0]
    return values


def reask(values: dict, fields) -> None:
    """Re-prompt for the parameters a cross-check complained about.

    The current value is offered as the default, so pressing enter keeps it and
    the user only has to change the one that is actually wrong.
    """
    spec = {name: (typ, sweep) for name, typ, _d, sweep, _h in PARAMS}
    for name in fields:
        typ, sweep = spec[name]
        current = values[name]
        shown = ",".join(str(v) for v in current) if isinstance(current, list) \
            else current
        parsed = ask(name, typ, shown, sweep)
        values[name] = parsed if sweep else parsed[0]


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def validate(values: dict) -> dict:
    """Cross-parameter checks -- the ones that need more than one value.

    Single values were already checked by VALIDATORS when they were entered.
    What is left are relationships: k and true_k against p, and whether beta*
    can actually be built at that p. Raises InputError naming the parameters to
    re-ask for, so an interactive caller can offer a correction instead of
    dying. Everything is checked before any run starts, so a typo in a swept
    list fails immediately rather than after the earlier combinations have
    already run.

    Returns the beta* vectors keyed by (preset, p, alpha), so main does not
    rebuild them.
    """
    if values["n_reps"] == 1:
        _warn("n_reps=1, so MSE sd and se are reported as 0")

    true_k = values["true_k"]
    betas, warned = {}, set()
    for p in values["p"]:
        for kk in values["k"]:
            if kk > p:
                raise InputError(f"k={kk} exceeds p={p}", ["k", "p"])
        if true_k > p:
            raise InputError(f"true_k={true_k} exceeds p={p}", ["true_k", "p"])

        for preset in values["beta"]:
            for a in values["alpha"]:
                kwargs = {"alpha": a} if preset == "weak_sparse" else {}
                try:
                    b = config.make_beta(preset, p, **kwargs)
                except ValueError as e:
                    raise InputError(f"beta preset {preset!r} at p={p}: {e}",
                                     ["beta", "p"]) from None
                betas[(preset, p, a)] = b

                # true_k is only used for scoring, so a mismatch silently scores
                # against the wrong support rather than failing.
                nnz = int(np.count_nonzero(b))
                if nnz != true_k and (preset, nnz) not in warned:
                    warned.add((preset, nnz))
                    _warn(f"beta preset {preset!r} at p={p} has {nnz} nonzeros but "
                          f"--true_k is {true_k}; top-k is scored against the {true_k} "
                          f"largest estimates, so it can never be 1 unless they match")

    for kk in values["k"]:
        if kk < true_k:
            _warn(f"--k {kk} is below --true_k {true_k}; k is meant to be an upper "
                  f"bound on the sparsity, so the algorithms cannot recover the "
                  f"full support")

    if any(v == 0 for v in values["sigma_a"]):
        _warn("sigma_a=0 means no measurement error, so Sigma_a=0 and the problem "
              "reduces to an ordinary lasso")

    # max_then_frobenius varies the norm along the homotopy, and only the
    # reweighted algorithms have one. For the single-pass ones it is the max
    # norm exactly, so sweeping both duplicates a run rather than adding one.
    if "max_then_frobenius" in values["norm"]:
        single_pass = [a for a in values["algorithm"] if a.startswith("cocolasso")]
        if single_pass and "max" in values["norm"]:
            _warn(f"norm 'max_then_frobenius' is identical to 'max' for "
                  f"{', '.join(sorted(set(single_pass)))} -- they run one iteration, "
                  f"so there is no later iteration to switch to frobenius; those "
                  f"combinations are duplicates")
        if not any(a.startswith("reweighted") for a in values["algorithm"]):
            _warn("norm 'max_then_frobenius' only differs from 'max' for the "
                  "reweighted algorithms, and none are being run")

    if any(str(v).lower() != "cv" and float(v) == 0 for v in values["lam"]):
        _warn("lam=0 leaves the regression unpenalised; with p > n this is not "
              "identifiable and the estimate will blow up")

    if any(str(v).lower() == "cv" for v in values["lam"]):
        fits = values["cv_folds"] * values["cv_lambdas"]
        _warn(f"lam='cv' costs {fits} extra fits per replication "
              f"({values['cv_folds']} folds x {values['cv_lambdas']} grid points); "
              f"reduce --cv_lambdas first if that is too slow")
        if "naive" in values["algorithm"]:
            raise InputError("algorithm 'naive' has no lambda to cross-validate",
                             ["algorithm", "lam"])

    return betas


def _tag(values: dict, name: str, limit: int = 3) -> str:
    """A filename-safe tag for one swept parameter."""
    uniq = list(dict.fromkeys(values[name]))
    if len(uniq) <= limit:
        return "+".join(str(v) for v in uniq)
    return f"{len(uniq)}{name}s"


def result_path(values: dict, now: datetime) -> Path:
    """results/<model>_<algorithm>_<MMDD>-<HHMM>.json

    Dropping the year and the seconds makes collisions between two runs in the
    same minute plausible, so a counter is appended rather than overwriting.
    """
    stem = (f"{_tag(values, 'model')}_{_tag(values, 'algorithm')}_"
            f"{now.strftime('%m%d-%H%M')}")
    path = RESULTS_DIR / f"{stem}.json"
    n = 2
    while path.exists():
        path = RESULTS_DIR / f"{stem}_{n}.json"
        n += 1
    return path


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    given = {name for name, *_ in PARAMS if getattr(args, name) is not None}
    interactive = args.interactive or not given

    values = collect(args, interactive)

    # Single values were validated as they were entered. Cross-parameter checks
    # can only run once everything is in, so on failure an interactive session
    # re-asks for the parameters involved instead of exiting.
    while True:
        try:
            betas = validate(values)
            break
        except InputError as e:
            print(f"\n  error: {e}", file=sys.stderr)
            if not interactive:
                return 2
            print("  correcting -- press enter to keep the current value.\n",
                  file=sys.stderr)
            reask(values, e.fields)

    sweep_names = [n for n in SWEEPABLE if n in values]
    sweep_names.sort(key=lambda n: [x[0] for x in PARAMS].index(n))
    combos = list(itertools.product(*(values[n] for n in sweep_names)))

    print(f"\n{len(combos)} run(s), {values['n_reps']} repetitions each, "
          f"seeds {values['seed']}..{values['seed'] + values['n_reps'] - 1}\n")

    runs = []
    header = (f"  {'model':>8} {'p':>5} {'sig_a':>6} {'lam':>5} {'norm':>10} "
              f"{'algorithm':>17} {'MSE':>9} {'MSE sd':>8} {'PE':>9} "
              f"{'top-k':>6} {'lam*':>7} {'s/fit':>8} {'total':>8}")
    if not args.quiet:
        print(header)
        print("  " + "-" * (len(header) - 2))

    for combo in combos:
        cv = dict(zip(sweep_names, combo))
        p = cv["p"]
        beta_star = betas[(cv["beta"], p, cv["alpha"])]  # built and checked in validate()
        algo_params = {"rho": cv["rho"]} if cv["algorithm"].startswith("reweighted") \
            else {}

        lam = cv["lam"] if str(cv["lam"]).lower() == "cv" else float(cv["lam"])

        def _report_rep(rep_index, seed, metrics, _n_reps=values["n_reps"],
                        _algo=cv["algorithm"], _norm=cv["norm"]):
            print(f"    [{_algo}/{_norm}] rep {rep_index + 1}/{_n_reps} "
                  f"(seed={seed}): MSE={metrics['mse']:.4f} PE={metrics['pe']:.4f} "
                  f"time={metrics['time']:.2f}s", flush=True)

        result = testmod.run_test(
            model=cv["model"], n=cv["n"], p=p, sigma_a=cv["sigma_a"],
            sigma_e=cv["sigma_e"], beta_star=beta_star, lam=lam,
            seeds=values["seed"], n_reps=values["n_reps"],
            algorithm=cv["algorithm"], norm=cv["norm"], k=cv["k"],
            true_k=values["true_k"], algo_params=algo_params,
            cv={"folds": values["cv_folds"], "n_lambdas": values["cv_lambdas"],
                "ratio": values["cv_ratio"]},
            progress=_report_rep if args.progress else None)
        result["config"]["beta_preset"] = cv["beta"]
        result["config"]["alpha"] = cv["alpha"]
        runs.append(result)

        if not args.quiet:
            m = result["metrics"]
            print(f"  {cv['model']:>8} {p:>5} {cv['sigma_a']:>6} {cv['lam']:>5} "
                  f"{cv['norm']:>10} {cv['algorithm']:>17} "
                  f"{m['mse_mean']:>9.4f} {m['mse_std']:>8.4f} "
                  f"{m['pe_mean']:>9.4f} {m['topk_rate']:>6.2f} "
                  f"{m['lam_mean']:>7.3f} "
                  f"{m['time_mean']:>8.3f} {result['runtime_sec']:>8.1f}",
                  flush=True)
            for what, items in result["ignored"].items():
                if items:
                    print(f"           note: {cv['algorithm']} ignores "
                          f"{what}: {', '.join(items)}", flush=True)

    now = datetime.now()
    out = Path(args.out) if args.out else result_path(values, now)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        # The filename drops the year and the seconds, so the full timestamp is
        # kept here where nothing is lost.
        {"timestamp": now.strftime("%Y%m%d-%H%M%S"),
         # Every parameter as resolved, whether the user gave it or it came
         # from a default, so the file alone reproduces the run.
         "parameters": values,
         "swept": sweep_names,
         "fixed": {name: values[name] for name, _t, _d, sweep, _h in PARAMS
                   if not sweep},
         "runs": runs}, indent=2), encoding="utf-8")
    print(f"\nsaved {len(runs)} run(s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
