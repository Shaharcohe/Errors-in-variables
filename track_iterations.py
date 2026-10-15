"""Per-iteration diagnostics for CoCoLasso and the reweighted homotopy.

One invocation = one (setting, algorithm, norm) over a range of seeds, at the
fixed parameters of the p=500 sweep (submit_sweep.sh). Data are generated
exactly as in test.run_test, so every final beta is bit-identical to the
corresponding sweep_p500 run.

    python track_iterations.py --model ar --sigma_a 1.0 --beta datta_zhang \\
        --algorithm reweighted --norm frobenius --seed 0 --n_reps 100 \\
        --out results/iteration_tracking/ar_sigma1.0_datta_zhang/reweighted_frobenius.json

At every iteration t, with S = the true_k largest |beta*_j| (for datta_zhang
exactly its support; weak_sparse has no exact zeros) and Sc its complement:

  cov_<block>_<norm>   Sigma_t - Sigma_X on block full / SS / SSc / ScSc,
                       in max and Frobenius norm. Sigma_t = C^-1/2 B_tilde C^-1/2
                       is what the regression at step t is solved against.
  beta_<part>_<norm>   beta_t - beta* on full / S / Sc, in l2 and l1.
  overlap_prev         |top-k(beta_t) & top-k(beta_{t-1})|, None at t = 0.
  top3_overlap         |top-s(beta_t) & top-s(beta*)|, s = true_k = 3.
  nnz                  nonzeros of beta_t.

CoCoLasso has a single iteration (t = 0). The same eight covariance errors of
the raw, unprojected Sigma_X_hat are stored once per repetition as `raw_cov`.

Writes the JSON at --out and the beta trajectories next to it (.npz, float32),
so new beta metrics can be computed later without refitting.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import config
import eiv_algorithm as eiv

# Fixed at the values of the p=500 sweep.
N, P, SIGMA_E, LAM, K, TRUE_K, RHO = 100, 500, 3.0, 1.0, 10, 3, 0.5
NZ_TOL = 1e-10


def top_k(beta, k):
    """Indices of the k largest |beta_j| among the nonzeros (fewer if nnz < k).

    Zeros are excluded rather than ranked: with nnz < k they would all tie, and
    which of them fill the remaining slots would be an argsort artefact.
    """
    nz = np.flatnonzero(np.abs(beta) > NZ_TOL)
    return nz[np.argsort(-np.abs(beta[nz]), kind="stable")[:k]]


def cov_errors(S, Sigma_x, supp, off):
    D = S - Sigma_x
    out = {}
    for name, block in (("full", D), ("SS", D[np.ix_(supp, supp)]),
                        ("SSc", D[np.ix_(supp, off)]),
                        ("ScSc", D[np.ix_(off, off)])):
        out[f"cov_{name}_max"] = float(np.abs(block).max())
        out[f"cov_{name}_fro"] = float(np.linalg.norm(block, "fro"))
    return out


def beta_errors(beta, beta_star, supp, off):
    d = beta - beta_star
    out = {}
    for name, part in (("full", d), ("S", d[supp]), ("Sc", d[off])):
        out[f"beta_{name}_l2"] = float(np.linalg.norm(part))
        out[f"beta_{name}_l1"] = float(np.abs(part).sum())
    return out


def run_tracked(algorithm, Z, y, Sigma_a, norm, Sigma_x, supp, off):
    """Fit, returning per-iteration covariance errors, betas and the info dict.

    Sigma_t is never formed by eiv_algorithm, so surrogate_design is wrapped to
    rebuild it from (B_tilde, c) as it passes through. Both estimators look the
    name up in the module at call time, so the wrapper sees every iteration.
    """
    n = Z.shape[0]
    covs, betas = [], []
    original = eiv.surrogate_design

    def recording_surrogate(B_tilde, c, Zty, n_):
        s = np.sqrt(c)
        covs.append(cov_errors(B_tilde / np.outer(s, s), Sigma_x, supp, off))
        return original(B_tilde, c, Zty, n_)

    def solver(Zt, yt, bp):                      # as in test._run_*, refit=False
        beta = eiv.solve_lasso(Zt, yt, n, LAM, beta_init=bp)
        betas.append(beta.copy())
        return beta

    eiv.surrogate_design = recording_surrogate
    try:
        if algorithm == "cocolasso":
            beta = eiv.cocolasso(Z, y, Sigma_a, projection=norm,
                                 regression_solver=solver)
            info = {"iters": 1, "converged": True}
        else:
            beta, info = eiv.reweighted_cocolasso(
                Z, y, Sigma_a, k=K, rho=RHO, projection=norm,
                regression_solver=solver, return_info=True)
    finally:
        eiv.surrogate_design = original
    assert len(covs) == len(betas) and np.array_equal(betas[-1], beta)
    return covs, betas, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(config.MODELS))
    ap.add_argument("--sigma_a", type=float, required=True)
    ap.add_argument("--beta", required=True, choices=sorted(config.BETA_PRESETS))
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weak_sparse decay exponent (ignored by other presets)")
    ap.add_argument("--algorithm", required=True, choices=["cocolasso", "reweighted"])
    ap.add_argument("--norm", required=True, choices=["max", "frobenius"])
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--n_reps", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    generate, sigma_x_fn = config.get_model(args.model)
    Sigma_x = sigma_x_fn(P)
    kwargs = {"alpha": args.alpha} if args.beta == "weak_sparse" else {}
    beta_star = config.make_beta(args.beta, P, **kwargs)
    supp = np.sort(np.argsort(-np.abs(beta_star), kind="stable")[:TRUE_K])
    off = np.setdiff1d(np.arange(P), supp)
    top_true = set(supp.tolist())
    Sigma_a = args.sigma_a ** 2 * np.eye(P)

    reps, all_betas = [], []
    for seed in range(args.seed, args.seed + args.n_reps):
        Z, y = generate(N, P, args.sigma_a, SIGMA_E, beta_star,
                        np.random.default_rng(seed))
        raw = cov_errors(Z.T @ Z / N - Sigma_a, Sigma_x, supp, off)
        t0 = time.perf_counter()
        covs, betas, info = run_tracked(args.algorithm, Z, y, Sigma_a, args.norm,
                                        Sigma_x, supp, off)
        runtime = time.perf_counter() - t0

        rows, prev = [], None
        for t, (cov, b) in enumerate(zip(covs, betas)):
            top = set(top_k(b, K).tolist())
            rows.append({"t": t, **cov, **beta_errors(b, beta_star, supp, off),
                         "overlap_prev": None if prev is None else len(top & prev),
                         "top3_overlap": len(set(top_k(b, TRUE_K).tolist()) & top_true),
                         "nnz": int(np.sum(np.abs(b) > NZ_TOL))})
            prev = top
        reps.append({"seed": seed, "iters": info["iters"],
                     "converged": bool(info["converged"]), "runtime": runtime,
                     "raw_cov": raw, "rows": rows})
        all_betas.append(np.array(betas, dtype=np.float32))
        last = rows[-1]
        print(f"seed {seed}: {info['iters']} iters, {runtime:.1f}s, "
              f"beta_l2 {last['beta_full_l2']:.3f}, top3 {last['top3_overlap']}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {"model": args.model, "n": N, "p": P, "sigma_a": args.sigma_a,
                   "sigma_e": SIGMA_E, "beta_preset": args.beta,
                   "alpha": args.alpha if args.beta == "weak_sparse" else None,
                   "lam": LAM, "k": K, "true_k": TRUE_K, "rho": RHO,
                   "algorithm": args.algorithm, "norm": args.norm,
                   "seeds": [r["seed"] for r in reps]},
        "support": supp.tolist(),
        "reps": reps,
    }
    # Write to a temporary name first: submit_tracking.sh skips any job whose
    # output exists, so a half-written file must never appear under the real name.
    for path, write in ((out.with_suffix(".npz"),
                         lambda f: np.savez_compressed(
                             f, beta=np.concatenate(all_betas),
                             iters=np.array([len(b) for b in all_betas]),
                             seeds=np.array(result["config"]["seeds"]))),
                        (out, lambda f: f.write(json.dumps(result).encode()))):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            write(fh)
        os.replace(tmp, path)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
