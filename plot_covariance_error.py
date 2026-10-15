"""Track the covariance estimate Sigma_t along the reweighted homotopy.

At iteration t the regression is solved against

    Sigma_t = C_t^-1/2 B_tilde_t C_t^-1/2,   B_tilde_t = proj(C_t^1/2 Sigma_X_hat C_t^1/2),

which reweighted_cocolasso never forms. This script captures (B_tilde_t, c_t)
by wrapping eiv_algorithm.surrogate_design, rebuilds Sigma_t, and plots its
distance to the true Sigma_X against t -- next to the raw bias-corrected
Sigma_X_hat, which is what every Sigma_t is a projection of.

Iteration 0 is plain CoCoLasso (uniform weights, C_0 a multiple of I), so the
first point of each curve is the CoCoLasso covariance.

    python plot_covariance_error.py                      main.py defaults, seed 0
    python plot_covariance_error.py --norm frobenius,max --seed 3
    python plot_covariance_error.py --p 500 --replot     redraw from the cache, no rerun

Each run caches its trajectories (results/covariance_error/cache_<tag>.pkl), so
the figures can be redrawn or extended without paying for the max-norm ADMM
again.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import eiv_algorithm as eiv

OUT_DIR = Path(__file__).resolve().parent / "results" / "covariance_error"


def run_tracked(Z, y, Sigma_a, k, lam, norm, rho):
    """reweighted_cocolasso, recording (Sigma_t, c_t, beta_t) at every t."""
    n = Z.shape[0]
    trace = []
    original = eiv.surrogate_design

    def recording_surrogate(B_tilde, c, Zty, n_):
        s = np.sqrt(c)
        trace.append({"Sigma": B_tilde / np.outer(s, s), "c": c.copy()})
        return original(B_tilde, c, Zty, n_)

    def solver(Zt, yt, bp):
        beta = eiv.solve_lasso(Zt, yt, n, lam, beta_init=bp)
        trace[-1]["beta"] = beta
        return beta

    eiv.surrogate_design = recording_surrogate
    try:
        beta, info = eiv.reweighted_cocolasso(
            Z, y, Sigma_a, k=k, rho=rho, projection=norm,
            regression_solver=solver, return_info=True)
    finally:
        eiv.surrogate_design = original
    return trace, info


def cov_errors(S, Sigma_x, supp):
    D = S - Sigma_x
    off = np.setdiff1d(np.arange(len(Sigma_x)), supp)
    return {
        "max": float(np.abs(D).max()),
        "fro": float(np.linalg.norm(D, "fro")),
        "op": float(np.linalg.norm(D, 2)),
        "SS_fro": float(np.linalg.norm(D[np.ix_(supp, supp)], "fro")),
        "ScS_max": float(np.abs(D[np.ix_(off, supp)]).max()),
        "lam_min": float(np.linalg.eigvalsh(0.5 * (S + S.T))[0]),
    }


def top_k(beta, k):
    """Indices of the k largest |beta_j| among the nonzeros (fewer if nnz < k).

    Zeros are excluded rather than ranked: with nnz < k they would all tie, and
    which of them fill the remaining slots would be an argsort artefact.
    """
    nz = np.flatnonzero(np.abs(beta) > 1e-10)
    return nz[np.argsort(-np.abs(beta[nz]), kind="stable")[:k]]


def rank_of(beta, j):
    """1-based rank of |beta_j| among all coordinates; nnz + 1 if beta_j = 0."""
    a = np.abs(beta)
    if a[j] <= 1e-10:
        return int(np.sum(a > 1e-10)) + 1
    return int(np.sum(a > a[j])) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ar")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=250)
    ap.add_argument("--sigma_a", type=float, default=1.0)
    ap.add_argument("--sigma_e", type=float, default=3.0)
    ap.add_argument("--beta", default="datta_zhang")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--norm", default="frobenius,max",
                    help="comma-separated projection schedules, all on the same data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the cached trajectories of a previous run")
    args = ap.parse_args()

    generate, sigma_x_fn = config.get_model(args.model)
    Sigma_x = sigma_x_fn(args.p)
    beta_star = config.make_beta(args.beta, args.p)
    supp = np.flatnonzero(beta_star)
    Z, y = generate(args.n, args.p, args.sigma_a, args.sigma_e, beta_star,
                    np.random.default_rng(args.seed))
    Sigma_a = args.sigma_a ** 2 * np.eye(args.p)
    Sigma_X_hat = Z.T @ Z / args.n - Sigma_a
    raw = cov_errors(Sigma_X_hat, Sigma_x, supp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.model}_n{args.n}_p{args.p}_sa{args.sigma_a}_{args.beta}"
           f"_lam{args.lam}_k{args.k}_rho{args.rho}_seed{args.seed}")
    cache = OUT_DIR / f"cache_{tag}.pkl"

    norms = [s.strip() for s in args.norm.split(",")]
    runs = {}
    if args.replot:
        with open(cache, "rb") as fh:
            runs = pickle.load(fh)
        print(f"loaded {cache} ({', '.join(runs)})")
        norms = []
    for norm in norms:
        print(f"running reweighted, norm={norm} ...", flush=True)
        trace, info = run_tracked(Z, y, Sigma_a, args.k, args.lam, norm, args.rho)
        rows = []
        for t, rec in enumerate(trace):
            d = rec["beta"] - beta_star
            c = rec["c"]
            rows.append({"t": t, **cov_errors(rec["Sigma"], Sigma_x, supp),
                         "beta_l2": float(np.linalg.norm(d)),
                         "pe": float(d @ Sigma_x @ d),
                         "c_supp": float(c[supp].mean()),
                         "c_off": float(np.delete(c, supp).mean())})
        runs[norm] = {"rows": rows, "info": info,
                      "F": np.array([1.0 - rec["c"] for rec in trace]),   # T x p, f_t
                      "B": np.array([rec["beta"] for rec in trace]),      # T x p, beta_t
                      "Sigma_first": trace[0]["Sigma"], "Sigma_last": trace[-1]["Sigma"]}
        r0, rT = rows[0], rows[-1]
        print(f"  {info['iters']} iterations (converged={info['converged']})\n"
              f"  ||Sigma_t - Sigma_X||_max : t=0 {r0['max']:.3f} -> t={rT['t']} {rT['max']:.3f}"
              f"   (raw Sigma_X_hat {raw['max']:.3f})\n"
              f"  ||Sigma_t - Sigma_X||_F   : t=0 {r0['fro']:.2f} -> t={rT['t']} {rT['fro']:.2f}"
              f"   (raw {raw['fro']:.2f})\n"
              f"  ||beta_t - beta*||_2      : t=0 {r0['beta_l2']:.3f} -> t={rT['t']} {rT['beta_l2']:.3f}",
              flush=True)
    if not args.replot:
        with open(cache, "wb") as fh:
            pickle.dump(runs, fh)

    setting = (f"{args.model.upper()}, n={args.n}, p={args.p}, $\\sigma_a$={args.sigma_a}, "
               f"$\\sigma_e$={args.sigma_e}, $\\beta^*$={args.beta}, $\\lambda$={args.lam}, "
               f"k={args.k}, $\\rho$={args.rho}, seed={args.seed}")
    colors = {"frobenius": "#2a78d6", "max": "#d9822b", "max_then_frobenius": "#3a9d5d"}

    # ---- Figure 1: error trajectories --------------------------------------
    panels = [
        ("max", r"$\|\Sigma_t-\Sigma_X\|_{\max}$", True),
        ("fro", r"$\|\Sigma_t-\Sigma_X\|_F$", True),
        ("op", r"$\|\Sigma_t-\Sigma_X\|_{op}$", True),
        ("SS_fro", r"support block $\|(\Sigma_t-\Sigma_X)_{SS}\|_F$", True),
        ("ScS_max", r"cross block $\|(\Sigma_t-\Sigma_X)_{S^cS}\|_{\max}$", True),
        ("lam_min", r"$\lambda_{\min}(\Sigma_t)$", False),
        ("beta_l2", r"$\|\beta_t-\beta^*\|_2$", False),
        ("pe", r"PE $(\beta_t-\beta^*)'\Sigma_X(\beta_t-\beta^*)$", False),
        ("c", r"mean $c_t = 1-f_t$: support (—) vs off (- -)", False),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), dpi=130)
    for ax, (key, title, show_raw) in zip(axes.ravel(), panels):
        for norm, run in runs.items():
            t = [r["t"] for r in run["rows"]]
            col = colors.get(norm, None)
            if key == "c":
                ax.plot(t, [r["c_supp"] for r in run["rows"]], "-o", ms=3, color=col, label=norm)
                ax.plot(t, [r["c_off"] for r in run["rows"]], "--", color=col)
            else:
                ax.plot(t, [r[key] for r in run["rows"]], "-o", ms=3, color=col, label=norm)
        if show_raw:
            ax.axhline(raw[key], color="#6b6a66", ls=":", lw=1.2,
                       label=r"raw $\hat\Sigma_X$ (unprojected)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("iteration t")
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Covariance error along the reweighted homotopy (t = 0 is CoCoLasso)\n"
                 + setting, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = OUT_DIR / f"trajectory_{tag}.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: entrywise error on the leading block ---------------------
    m = min(15, args.p)
    mats = [(r"raw $\hat\Sigma_X-\Sigma_X$", Sigma_X_hat)]
    for norm, run in runs.items():
        mats.append((f"{norm}, t=0", run["Sigma_first"]))
        mats.append((f"{norm}, t={run['rows'][-1]['t']} (final)", run["Sigma_last"]))
    diffs = [(lab, (S - Sigma_x)[:m, :m]) for lab, S in mats]
    vmax = max(np.abs(D).max() for _, D in diffs)
    fig, axes = plt.subplots(1, len(diffs), figsize=(3.6 * len(diffs), 3.9), dpi=130)
    axes = np.atleast_1d(axes)
    for ax, (lab, D) in zip(axes, diffs):
        im = ax.imshow(D, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(lab, fontsize=9)
        ax.set_xticks(supp[supp < m]); ax.set_yticks(supp[supp < m])
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label=r"$\Sigma-\Sigma_X$")
    fig.suptitle(f"Entrywise error, leading {m}x{m} block (ticks = true support)\n"
                 + setting, fontsize=10)
    p2 = OUT_DIR / f"heatmaps_{tag}.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: per-coordinate weights f_t --------------------------------
    # f_t is the damped GSM weight used at iteration t (built from beta_{t-1});
    # f = rho/p + (1 - rho) w, so the right axis reads off the GSM weight w.
    f_lo, f_hi = args.rho / args.p, args.rho / args.p + (1.0 - args.rho)
    supp_colors = ["#d03b3b", "#3a9d5d", "#8e44ad", "#e6a817", "#16a2b8"]
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 5), dpi=130,
                             squeeze=False)
    for ax, (norm, run) in zip(axes[0], runs.items()):
        F = run["F"]
        t = np.arange(len(F))
        mid = 0.5 * (f_lo + f_hi)
        selected = [j for j in np.flatnonzero(F[-1] < mid) if j not in supp]
        rest = np.setdiff1d(np.arange(args.p), np.concatenate([supp, selected]))
        ax.plot(t, F[:, rest], color="#b5b3ad", lw=0.6, alpha=0.5)
        ax.plot([], [], color="#b5b3ad", lw=0.8,
                label=f"other coordinates ({len(rest)})")
        for j in selected:
            ax.plot(t, F[:, j], "--", color="#2a78d6", lw=1.2)
        if selected:
            idx = [str(int(j)) for j in selected]
            lines = [", ".join(idx[i:i + 8]) for i in range(0, len(idx), 8)]
            ax.plot([], [], "--", color="#2a78d6",
                    label=f"kept, not in true support ({len(idx)}):\n" + ",\n".join(lines))
        for i, j in enumerate(supp):
            ax.plot(t, F[:, j], "-o", ms=3, lw=2, color=supp_colors[i % len(supp_colors)],
                    label=rf"true support $j$={j} ($\beta^*_j$={beta_star[j]:g})")
        for lvl in (f_lo, f_hi):
            ax.axhline(lvl, color="#6b6a66", ls=":", lw=1)
        ax.set_ylim(f_lo - 0.03, f_hi + 0.03)
        ax.set_xlabel("iteration t")
        ax.set_ylabel(r"weight $f_t(j)$  (→ low = kept, high = penalised)")
        sec = ax.secondary_yaxis("right", functions=(
            lambda f: (f - f_lo) / (1.0 - args.rho),
            lambda w: f_lo + (1.0 - args.rho) * w))
        sec.set_ylabel(r"GSM weight $w_t(j)$")
        ax.set_title(f"{norm} projection", fontsize=11, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False, loc="center left")
        for s in ("top",):
            ax.spines[s].set_visible(False)
    fig.suptitle("Per-coordinate weights along the homotopy\n" + setting, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p4 = OUT_DIR / f"weights_{tag}.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: support recovery ----------------------------------------
    # |supp(beta*) ∩ top-k(|beta_t|)| per iteration, and the rank of each true
    # support coordinate in |beta_t| -- the second shows *why* the first moves.
    hits = {nm: [len(set(top_k(b, args.k)) & set(supp.tolist())) for b in r["B"]]
            for nm, r in runs.items()}
    fig, axes = plt.subplots(1, 1 + len(runs), figsize=(6 * (1 + len(runs)), 4.6),
                             dpi=130, squeeze=False)
    ax = axes[0, 0]
    for i, (nm, h) in enumerate(hits.items()):
        off = 0.04 * (i - (len(hits) - 1) / 2)          # keep coincident steps visible
        ax.step(np.arange(len(h)), np.array(h) + off, where="mid", lw=2,
                color=colors.get(nm), label=nm)
        ax.plot(np.arange(len(h)), np.array(h) + off, "o", ms=4, color=colors.get(nm))
    ax.set_yticks(range(len(supp) + 1))
    ax.set_ylim(-0.3, len(supp) + 0.3)
    ax.set_xlabel("iteration t")
    ax.set_title(rf"true support in top-{args.k} of $|\beta_t|$  "
                 rf"(out of {len(supp)})", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    for ax, (nm, r) in zip(axes[0, 1:], runs.items()):
        B = r["B"]
        t = np.arange(len(B))
        for i, j in enumerate(supp):
            ax.plot(t, [rank_of(b, j) for b in B], "-o", ms=3, lw=2,
                    color=supp_colors[i % len(supp_colors)],
                    label=rf"$j$={j} ($\beta^*_j$={beta_star[j]:g})")
        nnz = [int(np.sum(np.abs(b) > 1e-10)) for b in B]
        ax.plot(t, np.array(nnz) + 1, ":", color="#6b6a66", lw=1.2,
                label=r"nnz$(\beta_t)$ + 1  (= rank of a zero)")
        ax.axhline(args.k + 0.5, color="#2a2a2a", ls="--", lw=1)
        ax.text(t[-1], args.k + 0.5, f" top-{args.k} cutoff", va="bottom",
                ha="right", fontsize=8)
        ax.set_yscale("log")
        ticks = [v for v in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500)
                 if v <= max(nnz) + 1 or v <= args.k]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(v) for v in ticks])
        ax.minorticks_off()
        ax.invert_yaxis()
        ax.set_xlabel("iteration t")
        ax.set_ylabel(r"rank of $|\beta_t(j)|$  (1 = largest)")
        ax.set_title(f"{nm} projection: rank of each true coordinate",
                     fontsize=10, loc="left")
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[0]:
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Support recovery along the homotopy\n" + setting, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p5 = OUT_DIR / f"support_{tag}.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    for nm, h in hits.items():
        print(f"  {nm}: |supp ∩ top-{args.k}| by iteration = {h}")

    p3 = OUT_DIR / f"trajectory_{tag}.json"
    p3.write_text(json.dumps({"args": vars(args), "raw": raw,
                              "runs": {nm: {"info": r["info"], "rows": r["rows"],
                                            "f": r["F"].tolist(),
                                            "beta": r["B"].tolist(),
                                            "support_hits": hits[nm]}
                                       for nm, r in runs.items()}},
                             indent=1, default=float))
    print(f"saved {p1}\nsaved {p2}\nsaved {p4}\nsaved {p5}\nsaved {p3}")


if __name__ == "__main__":
    main()
