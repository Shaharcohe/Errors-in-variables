"""Run one experimental configuration over several repetitions.

run_test() takes a full specification -- model, data parameters, lambda, seeds,
algorithm, norm, k -- and returns the per-repetition metrics together with their
aggregates. It holds no defaults of its own and prints nothing; main.py supplies
the defaults and does the reporting.

The algorithms are thin wrappers over eiv_algorithm.py. Two points about how
they consume their arguments, both of which caused misleading comparisons
before they were made explicit:

  * k means different things. For 'reweighted' it is a target: the GSM weights
    try to hold exactly k coordinates active. For the refit solvers it is only a
    cap on the lasso's own nonzeros, so a k above the selected support size is
    inert. A refit handed the true k looks far stronger than one handed a loose
    upper bound, which is the realistic case.
  * not every algorithm uses every argument. 'naive' ignores lam and norm
    entirely; 'cocolasso' and 'cocolasso_refit' ignore every reweighting
    parameter. run_test records the ignored arguments in its result rather than
    accepting them silently.

true_k is used only for scoring and is never passed to an algorithm.
"""
from __future__ import annotations

import inspect
import time
import numpy as np

import config
from eiv_algorithm import (cocolasso, reweighted_cocolasso, naive_topk,
                           solve_lasso, make_refit_solver, select_lambda_cv)

# ----------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------

# Parameters each algorithm accepts from algo_params; anything else passed in is
# reported as ignored.
_ACCEPTS = {
    "cocolasso": {"admm_rho", "admm_max_iter", "psd_floor"},
    "cocolasso_refit": {"admm_rho", "admm_max_iter", "psd_floor", "refit_thresh"},
    "reweighted": {"rho", "gamma_rule", "gamma1", "growth", "gamma_schedule",
                   "f_tol", "min_iter", "max_iter", "weight_clip",
                   "admm_rho", "admm_max_iter", "psd_floor"},
    "reweighted_refit": {"rho", "gamma_rule", "gamma1", "growth", "gamma_schedule",
                         "f_tol", "min_iter", "max_iter", "weight_clip",
                         "admm_rho", "admm_max_iter", "psd_floor", "refit_thresh"},
    "naive": {"psd_floor"},
}

# Arguments of run_test itself that an algorithm does not use.
_UNUSED_ARGS = {
    "cocolasso": ("k",),
    "cocolasso_refit": (),
    "reweighted": (),
    "reweighted_refit": (),
    "naive": ("lam", "norm"),
}


# The underlying functions each algorithm reaches, so their signature defaults
# can be reported alongside whatever the caller overrode.
_UNDERLYING = {
    "cocolasso": (cocolasso,),
    "cocolasso_refit": (cocolasso, make_refit_solver),
    "reweighted": (reweighted_cocolasso,),
    "reweighted_refit": (reweighted_cocolasso, make_refit_solver),
    "naive": (naive_topk,),
}


def effective_algo_params(algorithm: str, given: dict) -> dict:
    """Every parameter the algorithm actually ran with, defaults included.

    Recording only what the caller passed leaves the run under-specified: a
    result saying algo_params={} does not say which psd_floor or which gamma
    schedule produced it, and those defaults can change. This reads the
    defaults off the signatures of the functions the algorithm calls and
    overlays the overrides.
    """
    out = {}
    for fn in _UNDERLYING[algorithm]:
        for name, prm in inspect.signature(fn).parameters.items():
            if name in _ACCEPTS[algorithm] and prm.default is not prm.empty:
                out[name] = prm.default
    out.update({kk: vv for kk, vv in given.items() if kk in _ACCEPTS[algorithm]})
    return out


def _run_cocolasso(Z, y, Sigma_a, lam, norm, k, n, params, refit):
    solver = (make_refit_solver(lam, n, k=k,
                                thresh=params.pop("refit_thresh", 1e-6))
              if refit else
              (lambda Zt, yt, bp: solve_lasso(Zt, yt, n, lam, beta_init=bp)))
    return cocolasso(Z, y, Sigma_a, projection=norm,
                     regression_solver=solver, **params), {}


def _run_reweighted(Z, y, Sigma_a, lam, norm, k, n, params, refit):
    solver = (make_refit_solver(lam, n, k=k,
                                thresh=params.pop("refit_thresh", 1e-6))
              if refit else
              (lambda Zt, yt, bp: solve_lasso(Zt, yt, n, lam, beta_init=bp)))
    beta, info = reweighted_cocolasso(Z, y, Sigma_a, k=k, projection=norm,
                                      regression_solver=solver,
                                      return_info=True, **params)
    return beta, info


ALGORITHMS = {
    "cocolasso": lambda *a: _run_cocolasso(*a, refit=False),
    "cocolasso_refit": lambda *a: _run_cocolasso(*a, refit=True),
    "reweighted": lambda *a: _run_reweighted(*a, refit=False),
    "reweighted_refit": lambda *a: _run_reweighted(*a, refit=True),
    "naive": lambda Z, y, Sigma_a, lam, norm, k, n, params: (
        naive_topk(Z, y, Sigma_a, k, **params), {}),
}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def compute_metrics(beta_hat: np.ndarray, beta_star: np.ndarray,
                    sigma_x: np.ndarray, true_k: int) -> dict:
    """MSE, prediction error and exact top-k recovery for one estimate.

    PE is the population prediction error E[(x'beta_hat - x'beta*)^2] over a
    fresh x ~ N(0, Sigma_x). It is exact given Sigma_x, so unlike a held-out
    test sample it adds no Monte-Carlo noise of its own, and it is measured on
    the true covariates rather than the noisy Z.
    """
    d = np.asarray(beta_hat, dtype=float) - np.asarray(beta_star, dtype=float)
    top = set(np.argsort(-np.abs(beta_hat))[:true_k].tolist())
    return {
        "mse": float(d @ d),
        "pe": float(d @ sigma_x @ d),
        "topk": float(top == set(np.flatnonzero(beta_star).tolist())),
        "nnz": int(np.sum(np.abs(beta_hat) > 1e-6)),
    }


# ----------------------------------------------------------------------
# The test driver
# ----------------------------------------------------------------------

def resolve_seeds(seeds, n_reps: int) -> list:
    """One seed per repetition; a bare int expands to seed, seed+1, ..."""
    if seeds is None:
        return list(range(n_reps))
    if np.isscalar(seeds):
        return [int(seeds) + i for i in range(n_reps)]
    seeds = list(seeds)
    if len(seeds) != n_reps:
        raise ValueError(f"got {len(seeds)} seeds for {n_reps} repetitions")
    return [int(s) for s in seeds]


def run_test(model: str, n: int, p: int, sigma_a: float, sigma_e: float,
             beta_star: np.ndarray, lam, seeds, n_reps: int,
             algorithm: str, norm: str, k: int, true_k: int,
             algo_params: dict | None = None,
             cv: dict | None = None) -> dict:
    """Run `algorithm` on `n_reps` datasets from `model` and score it.

    lam may be a number, or the string "cv" to choose it per replication by
    corrected k-fold cross-validation (Datta & Zhang, Sec. 6.2). Selection is
    redone on every replication because it is part of the estimator: reusing
    one lambda across replications would understate the estimator's variance.
    cv holds {"folds", "n_lambdas", "ratio"}.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; "
                         f"available: {sorted(ALGORITHMS)}")
    if norm not in ("max", "frobenius"):
        raise ValueError(f"norm must be 'max' or 'frobenius', got {norm!r}")

    use_cv = isinstance(lam, str) and lam.lower() == "cv"
    if use_cv and algorithm == "naive":
        raise ValueError("algorithm 'naive' has no lambda to cross-validate")
    cv = {"folds": 5, "n_lambdas": 20, "ratio": 1e-2, **(cv or {})}

    generate, sigma_x_fn = config.get_model(model)
    sigma_x = sigma_x_fn(p)
    beta_star = np.asarray(beta_star, dtype=float)
    seed_list = resolve_seeds(seeds, n_reps)

    algo_params = dict(algo_params or {})
    accepted = _ACCEPTS[algorithm]
    ignored_params = sorted(set(algo_params) - accepted)
    params = {kk: vv for kk, vv in algo_params.items() if kk in accepted}

    Sigma_a = sigma_a ** 2 * np.eye(p)
    per_rep, info_last, t0 = [], {}, time.perf_counter()
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        Z, y = generate(n, p, sigma_a, sigma_e, beta_star, rng)

        # Timed around the fit only: data generation is common to every
        # algorithm, so including it would flatter the slow ones. When lambda
        # is cross-validated the selection is inside the timing, because it is
        # part of what the estimator costs.
        t_fit = time.perf_counter()
        if use_cv:
            def refit(Ztr, ytr, Sa, lm, _n=None):
                return ALGORITHMS[algorithm](
                    Ztr, ytr, Sa, lm, norm, k, len(ytr), dict(params))[0]
            lam_used, cv_losses = select_lambda_cv(
                Z, y, Sigma_a, refit, folds=cv["folds"], seed=seed,
                n_lambdas=cv["n_lambdas"], ratio=cv["ratio"])
        else:
            lam_used, cv_losses = float(lam), None
        beta_hat, info = ALGORITHMS[algorithm](
            Z, y, Sigma_a, lam_used, norm, k, n, dict(params))
        elapsed = time.perf_counter() - t_fit

        metrics = compute_metrics(beta_hat, beta_star, sigma_x, true_k)
        metrics["time"] = elapsed
        metrics["lam"] = lam_used
        per_rep.append(metrics)
        info_last = info
        if cv_losses is not None:
            info_last = {**info, "cv_losses": np.asarray(cv_losses).tolist()}

    def col(name):
        return np.array([r[name] for r in per_rep], dtype=float)

    mse, pe, tm = col("mse"), col("pe"), col("time")
    return {
        "config": {
            "model": model, "n": n, "p": p, "sigma_a": sigma_a,
            "sigma_e": sigma_e,
            # lam as asked for ("cv" or a number) and as actually used on each
            # replication, so a cross-validated run still records its lambdas.
            "lam": lam,
            "lam_requested": lam,
            "lam_selected": [float(r["lam"]) for r in per_rep],
            "cv": cv if use_cv else None,
            "algorithm": algorithm,
            "norm": norm, "k": k, "true_k": true_k, "n_reps": n_reps,
            "seeds": seed_list,
            "algo_params": effective_algo_params(algorithm, params),
            "algo_params_given": params,
            "beta_star_nonzeros": {int(i): float(beta_star[i])
                                   for i in np.flatnonzero(beta_star)},
            "snr": config.snr(model, p, beta_star, sigma_e),
        },
        "ignored": {
            "algo_params": ignored_params,
            "arguments": list(_UNUSED_ARGS[algorithm]),
        },
        "metrics": {
            "mse_mean": float(mse.mean()),
            "mse_std": float(mse.std(ddof=1)) if n_reps > 1 else 0.0,
            "mse_se": float(mse.std(ddof=1) / np.sqrt(n_reps)) if n_reps > 1 else 0.0,
            "pe_mean": float(pe.mean()),
            "pe_std": float(pe.std(ddof=1)) if n_reps > 1 else 0.0,
            "pe_se": float(pe.std(ddof=1) / np.sqrt(n_reps)) if n_reps > 1 else 0.0,
            "topk_rate": float(col("topk").mean()),
            "nnz_mean": float(col("nnz").mean()),
            "time_mean": float(tm.mean()),
            "time_std": float(tm.std(ddof=1)) if n_reps > 1 else 0.0,
            "time_total": float(tm.sum()),
            # Under CV this varies by replication; with a fixed lambda the
            # spread is zero and the mean is simply that lambda.
            "lam_mean": float(col("lam").mean()),
            "lam_std": float(col("lam").std(ddof=1)) if n_reps > 1 else 0.0,
        },
        "per_rep": {name: col(name).tolist()
                    for name in ("mse", "pe", "topk", "nnz", "time", "lam")},
        # Wall clock for the whole call, so it includes data generation and
        # scoring; time_total above covers the fits alone.
        "runtime_sec": time.perf_counter() - t0,
        "homotopy_iters": info_last.get("iters"),
    }
