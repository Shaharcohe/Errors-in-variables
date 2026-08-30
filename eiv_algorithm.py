"""
Reweighted Covariance-Corrected Sparse Regression
Combining the CoCoLasso projection with the Generalized Soft-Min (GSM) penalty.

Reference:
    T. Amir, R. Basri, B. Nadler, "The Trimmed Lasso: Sparse Recovery Guarantees
    and Practical Optimization by the Generalized Soft-Min Penalty",
    SIAM J. Math. Data Sci. 3(3):900-929, 2021.
    A. Datta, C.-H. Zhang, "CoCoLasso for High-Dimensional Error-in-Variables
    Regression", Ann. Statist. 45(6):2400-2426, 2017.

Model: y = X beta* + eps, observed Z = X + A with known noise covariance Sigma_a.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from sparse_approx_gsm import gsm

# ----------------------------------------------------------------------
# Section 2: Generalized Soft-Min (GSM) weights
# ----------------------------------------------------------------------

def gsm_weights(beta: np.ndarray, k: int, gamma: float) -> np.ndarray:
    """GSM weight vector w_{k,gamma}(beta), eq. (5)/(7).

    w_i = P[coordinate i excluded from the top-k support] in [0, 1].

    This delegates directly to the original sparse-approx-gsm implementation.
    """
    beta = np.asarray(beta, dtype=float)
    p = beta.shape[0]
    _, weights = gsm(np.abs(beta), p - k, -gamma)
    return weights


def damped_weights(beta: np.ndarray, k: int, gamma: float, rho: float) -> np.ndarray:
    """Damped GSM weights f^(gamma)(beta), eq. (8)."""
    p = beta.shape[0]
    w = gsm_weights(beta, k, gamma)
    return rho / p + (1.0 - rho) * w


# ----------------------------------------------------------------------
# The gamma schedule of Amir, Basri and Nadler
#
# Ported from the reference implementation (sparse_approx_gsm.solver), whose
# schedule differs from a plain geometric grid in three ways that matter:
#
#   * gamma_1 is set relative to the spread of |beta|, not as an absolute
#     constant, so the path does not depend on the scale of the data;
#   * the growth factor is tiny (1.02), but gamma keeps being multiplied until
#     the weights actually move by a set amount -- the step size is controlled
#     by weight movement rather than by gamma itself. A geometric x2 grid jumps
#     through the entire transition in one or two steps;
#   * the path ends at gamma = infinity exactly, once the weights have
#     saturated, rather than at a large finite gamma.
#
# The reference sets i_growth = 0 and never increments it, so only the first of
# its four growth factors is ever used; that single factor is reproduced here.
# ----------------------------------------------------------------------

GAMMA_FIRST_MAX_DIFF = 1e-4      # gamma_first_max_difference_of_sums
GAMMA_GROWTH = 1.02              # gamma_growth_factors[0], 'normal' preset
W_DIFF_THRESH = 1e-3             # w_diff_thresh_to_keep_increasing_gamma
W_SPARSITY_THRESH = 1e-5         # sparsity_thresh_abs_w
N_SPARSE_W_TO_STOP = 2           # n_gammas_sparse_w_to_stop


def max_diff_of_sums(beta: np.ndarray, k: int) -> float:
    """Delta_k(|beta|), the spread used to set gamma_1."""
    z = np.sort(np.abs(beta))[::-1]
    d = z.size
    if k >= d:
        return 0.0
    top, bot = z[:d - k], z[k:]
    return float(np.max(np.abs(top - bot))) if top.size > 0 else 0.0


def weight_nonsparsity(w: np.ndarray, k: int) -> float:
    """Largest weight among the k coordinates that should carry weight 0.

    w has p - k entries near 1 and k near 0 once saturated, so this is the
    (p-k+1)-th largest entry and it going to 0 is the saturation test.
    """
    p = w.size
    if p - k >= p:
        return 0.0
    return float(-np.partition(-np.abs(w), p - k)[p - k])


def next_gamma_amir(gamma: float, beta: np.ndarray, k: int,
                    growth: float = GAMMA_GROWTH,
                    w_diff_thresh: float = W_DIFF_THRESH) -> float:
    """Advance gamma until the GSM weights move by a set amount."""
    p = beta.shape[0]
    w_curr = gsm_weights(beta, k, gamma)
    target = (p - k) / p * w_diff_thresh
    gamma_next = gamma * growth
    for _ in range(100):                       # safety limit, as in the reference
        if np.abs(gsm_weights(beta, k, gamma_next) - w_curr).max() > target:
            break
        gamma_next *= growth
    return gamma_next


# ----------------------------------------------------------------------
# Section 3.2: CoCoLasso weighted PSD projection
# ----------------------------------------------------------------------

def _project_l1_ball(v: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection of v onto {x : ||x||_1 <= radius} (Duchi et al. 2008)."""
    if radius <= 0:
        return np.zeros_like(v)
    u = np.abs(v)
    if u.sum() <= radius:
        return v.copy()
    s = np.sort(u)[::-1]
    css = np.cumsum(s)
    idx = np.arange(1, len(v) + 1)
    cond = s - (css - radius) / idx > 0
    rho_idx = idx[cond][-1]
    theta = (css[rho_idx - 1] - radius) / rho_idx
    return np.sign(v) * np.maximum(u - theta, 0.0)


def _prox_linf(v: np.ndarray, t: float) -> np.ndarray:
    """Proximal operator of t * ||.||_inf (max abs entry), via Moreau decomposition."""
    if t <= 0:
        return v.copy()
    return v - t * _project_l1_ball(v / t, 1.0)


def _project_psd(M: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Frobenius projection of a symmetric matrix onto {B : B >= floor * I}."""
    Ms = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(Ms)
    eigvals = np.clip(eigvals, floor, None)
    return (eigvecs * eigvals) @ eigvecs.T


def frobenius_projection(M: np.ndarray, psd_floor: float = 0.0) -> np.ndarray:
    """Projection onto {B >= psd_floor * I} in Frobenius norm -- closed form.

    One eigendecomposition with the eigenvalues clipped at psd_floor, against the
    50-300 ADMM iterations (each containing its own eigendecomposition) that the
    max-norm projection needs. That is the whole appeal: it makes a long gamma
    homotopy affordable.

    What it gives up is the reason Datta & Zhang chose the max-norm in the first
    place. There, Sigma_X being feasible gives

        ||Sigma_tilde - Sigma_X||_max <= 2 ||Sigma_hat_X - Sigma_X||_max,

    so the entrywise deviation that drives the lasso analysis survives the
    projection with only a factor of 2. The same argument in Frobenius norm
    bounds ||.||_F, and converting back costs a factor of p:

        ||.||_max <= ||.||_F <= 2 ||Sigma_hat_X - Sigma_X||_F
                            <= 2p ||Sigma_hat_X - Sigma_X||_max.

    So this variant carries no rate guarantee. Whether that matters in practice
    is exactly what it is here to test.
    """
    return _project_psd(M, floor=psd_floor)


def project_corrected(M: np.ndarray, psd_floor: float = 0.0,
                      projection: str = "max", admm_rho: float = 1.0,
                      admm_max_iter: int = 300) -> np.ndarray:
    """Dispatch to the max-norm (ADMM) or Frobenius (closed-form) projection."""
    if projection == "frobenius":
        return frobenius_projection(M, psd_floor=psd_floor)
    if projection == "max":
        return cocolasso_projection(M, psd_floor=psd_floor, admm_rho=admm_rho,
                                    max_iter=admm_max_iter)
    raise ValueError(f"projection must be 'max' or 'frobenius', got {projection!r}")


def cocolasso_projection(M: np.ndarray, psd_floor: float = 0.0, admm_rho: float = 1.0,
                          max_iter: int = 300, tol: float = 1e-8) -> np.ndarray:
    """Solve argmin_{B >= psd_floor * I} ||M - B||_max via ADMM (Datta & Zhang,
    2017, Sec 3.2).

    Splits B (cone-constrained copy) from Delta = M - B (max-norm penalized), and
    alternates a PSD projection with a proximal ell-infinity step.

    psd_floor > 0 replaces the PSD cone {B >= 0} by the closed convex subset
    {B >= psd_floor * I}, which keeps B_tilde strictly positive definite so that
    it admits a Cholesky factor (see surrogate_design). With psd_floor = 0 the
    minimiser necessarily lies on the boundary of the cone whenever M is not
    already PSD -- moving from any interior B towards M stays feasible and
    strictly decreases ||M - B||_max -- hence B_tilde is singular there. The
    max-norm guarantee ||B_tilde - Sigma_X||_max <= 2 ||M - Sigma_X||_max only
    needs the target to be feasible, so it survives as long as
    Sigma_X >= psd_floor * I.
    """
    M = 0.5 * (M + M.T)
    p = M.shape[0]
    B = M.copy()
    Z = M.copy()
    U = np.zeros_like(M)
    for _ in range(max_iter):
        B_new = _project_psd(Z - U, floor=psd_floor)
        W = B_new + U
        delta = _prox_linf((M - W).ravel(), 1.0 / admm_rho).reshape(p, p)
        delta = 0.5 * (delta + delta.T)
        Z_new = M - delta
        U = U + B_new - Z_new
        if (np.linalg.norm(B_new - B) < tol * (1 + np.linalg.norm(B)) and
                np.linalg.norm(Z_new - Z) < tol * (1 + np.linalg.norm(Z))):
            B, Z = B_new, Z_new
            break
        B, Z = B_new, Z_new
    return 0.5 * (B + B.T)


# ----------------------------------------------------------------------
# Section 3.3: surrogate data and the regression step
# ----------------------------------------------------------------------

def surrogate_design(B_tilde: np.ndarray, c: np.ndarray, Zty: np.ndarray,
                     n: int) -> tuple:
    """Clean surrogate data (Z_tilde, y_tilde) for the Step-8 subproblem.

    Datta & Zhang replace the quadratic form by a least-squares fit on a
    manufactured data set characterised by

        Z_tilde^T Z_tilde / n = Sigma_t,      Z_tilde^T y_tilde = Z^T y,

    so that  1/(2n) ||y_tilde - Z_tilde beta||^2  =  1/2 beta^T Sigma_t beta
    - <Z^T y / n, beta> + const, and any regression solver can take over.

    Here Sigma_t = C^-1/2 B_tilde C^-1/2 is never formed. Factoring B_tilde
    instead, B_tilde = L L^T, gives Sigma_t = (C^-1/2 L)(C^-1/2 L)^T directly,
    so the Cholesky factor of Sigma_t is L^T C^-1/2 and the only inverse weight
    left is a column scaling of a triangular matrix:

        Z_tilde = sqrt(n) * L^T C^-1/2,   y_tilde = L^-1 C^1/2 Z^T y / sqrt(n).

    B_tilde must be positive definite; see the psd_floor argument of
    cocolasso_projection. Z_tilde is then upper triangular, which
    solve_unpenalized exploits.
    """
    c_sqrt = np.sqrt(c)
    L = np.linalg.cholesky(0.5 * (B_tilde + B_tilde.T))

    Z_tilde = np.sqrt(n) * (L.T / c_sqrt[None, :])
    y_tilde = solve_triangular(L, c_sqrt * Zty, lower=True) / np.sqrt(n)
    return Z_tilde, y_tilde


def solve_unpenalized(Z_tilde: np.ndarray, y_tilde: np.ndarray,
                      beta_init: np.ndarray | None = None) -> np.ndarray:
    """Default R = 0: minimise 1/(2n) ||y_tilde - Z_tilde beta||^2.

    Equivalently the normal equations Sigma_t beta = Z^T y / n. Since Z_tilde
    is square, upper triangular and non-singular (psd_floor > 0), the least
    squares solution is the single back-substitution Z_tilde beta = y_tilde,
    in O(p^2) and without forming Sigma_t. beta_init is accepted so that the
    signature matches the regression_solver protocol; it is unused.
    """
    return solve_triangular(Z_tilde, y_tilde, lower=False)


def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def solve_lasso(Z_tilde: np.ndarray, y_tilde: np.ndarray, n: int, lam: float,
                beta_init: np.ndarray | None = None,
                max_iter: int = 20000, tol: float = 1e-10) -> np.ndarray:
    """Solve argmin_beta 1/(2n) ||y_tilde - Z_tilde beta||^2 + lam ||beta||_1
    by accelerated proximal gradient (FISTA), for R(beta) = lam * ||beta||_1.

    Not the default; pass it explicitly, e.g.
        regression_solver=lambda Zt, yt, b: solve_lasso(Zt, yt, n, 0.15, b)

    Acceleration matters here: cond(Sigma_t) is of order 1 / psd_floor, since
    the floor sets the smallest eigenvalue while the largest is O(1). Plain
    ISTA needs O(cond) iterations, FISTA O(sqrt(cond)).
    """
    p = Z_tilde.shape[1]
    beta = np.zeros(p) if beta_init is None else beta_init.copy()
    L = max(np.linalg.norm(Z_tilde, 2) ** 2 / n, 1e-8)
    step = 1.0 / L

    z = beta.copy()
    theta = 1.0
    for _ in range(max_iter):
        grad = Z_tilde.T @ (Z_tilde @ z - y_tilde) / n
        beta_new = _soft_threshold(z - step * grad, step * lam)
        theta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta ** 2))
        z = beta_new + ((theta - 1.0) / theta_new) * (beta_new - beta)
        converged = np.linalg.norm(beta_new - beta) < tol * (1 + np.linalg.norm(beta))
        beta, theta = beta_new, theta_new
        if converged:
            break
    return beta


def refit_on_support(Z_tilde: np.ndarray, y_tilde: np.ndarray,
                     support: np.ndarray) -> np.ndarray:
    """Unpenalized least squares restricted to `support`, zero elsewhere.

    On the surrogate this is argmin over beta supported on S of
    ||y_tilde - Z_tilde beta||^2, i.e. beta_S = Sigma_t[S, S]^-1 (Z^T y / n)_S.
    """
    p = Z_tilde.shape[1]
    beta = np.zeros(p)
    if len(support) == 0:
        return beta
    beta[support] = np.linalg.lstsq(Z_tilde[:, support], y_tilde, rcond=None)[0]
    return beta


def refit_on_support_raw(Sigma_X_hat: np.ndarray, Zty: np.ndarray, n: int,
                         support: np.ndarray, psd_floor: float = 1e-6,
                         admm_max_iter: int = 300, ridge: float = 1.0) -> np.ndarray:
    """Ridge-penalized refit on `support` from the bias-corrected covariance,
    not the projected one.

    Solves (Sigma[S, S] + ridge * scale * I) beta_S = (Z^T y / n)_S where
    Sigma[S, S] is the |S| x |S| block of Sigma-hat_X, projected onto the PSD
    cone only if it is not already there, and scale = trace(block)/|S|.
    Contrast refit_on_support, which uses the block of the *globally* projected
    Sigma_t: that projection minimises the max-norm error over all p^2 entries
    and so spends accuracy away from the block we care about, whereas for
    |S| << n the raw block is a more targeted estimate of Sigma_X[S, S].

    The ridge term is unconditional, not just a fallback when the PSD check
    fails: that check only catches a negative eigenvalue, not poor
    conditioning -- a tiny positive eigenvalue passes it and then blows up
    np.linalg.solve. ridge=0 reproduces exactly that failure, which is what
    the unpenalized version of this refit did at higher measurement noise.
    """
    p = Sigma_X_hat.shape[0]
    beta = np.zeros(p)
    if len(support) == 0:
        return beta

    block = Sigma_X_hat[np.ix_(support, support)]
    scale = max(abs(np.trace(block)) / len(support), 1e-12)
    floor = psd_floor * scale
    if np.linalg.eigvalsh(0.5 * (block + block.T))[0] < floor:
        block = cocolasso_projection(block, psd_floor=floor,
                                     max_iter=admm_max_iter)
    block = block + (ridge * scale) * np.eye(len(support))
    beta[support] = np.linalg.solve(block, Zty[support] / n)
    return beta


def make_refit_solver(lam: float, n: int, k: int | None = None,
                      thresh: float = 1e-6, Sigma_X_hat: np.ndarray | None = None,
                      Zty: np.ndarray | None = None, ridge: float = 1.0):
    """Lasso, then a debiasing refit on the coordinates it selected.

    The lasso shrinks the true coefficients towards zero; refitting on the
    selected support with little or no penalty removes most of that shrinkage
    while keeping the selection. This is the cheap debiasing baseline that any
    reweighting scheme has to beat, since reweighting produces a similar effect
    at T times the cost.

    Support rule: the top-k coordinates if k is given (matching the information
    reweighted_cocolasso is handed), otherwise every nonzero of the lasso fit.

    Refit variant: by default the block of the already-projected Sigma_t, read
    off Z_tilde for free, refit unpenalized (that block is well-conditioned by
    construction, since it came out of the global PSD projection). Pass
    Sigma_X_hat and Zty to refit from the raw bias-corrected block instead (see
    refit_on_support_raw); that block is not guaranteed well-conditioned, so it
    is refit with a small ridge penalty (see `ridge`) rather than exactly --
    exact refitting there is what produced the numerical breakdowns at higher
    measurement noise that motivated the ridge term.

    Returns a callable matching the regression_solver protocol, so it plugs into
    either cocolasso() or reweighted_cocolasso().
    """
    raw = Sigma_X_hat is not None and Zty is not None

    def solver(Z_tilde, y_tilde, beta_prev):
        b = solve_lasso(Z_tilde, y_tilde, n, lam, beta_init=beta_prev)
        nz = np.flatnonzero(np.abs(b) > thresh)
        if k is not None:
            support = nz[np.argsort(-np.abs(b[nz]))[:k]]
        else:
            support = nz
        if raw:
            return refit_on_support_raw(Sigma_X_hat, Zty, n, support, ridge=ridge)
        return refit_on_support(Z_tilde, y_tilde, support)
    return solver


# ----------------------------------------------------------------------
# Baseline: plain CoCoLasso (Datta & Zhang, 2017)
# ----------------------------------------------------------------------

def naive_topk(Z: np.ndarray, y: np.ndarray, Sigma_a: np.ndarray, k: int,
               psd_floor: float = 1e-6) -> np.ndarray:
    """The naive baseline: marginal screening, then an unpenalized fit on the k
    survivors.

    Rank the coordinates by |Z_j^T y| / ||Z_j|| (marginal correlation, sure
    independence screening), keep the top k, and solve
    Sigma-hat_X[S, S] beta_S = (Z^T y / n)_S, projecting that k x k block onto
    the PSD cone only if it is not already there.

    No lasso, no homotopy, and -- crucially -- no p x p projection: the only
    matrix ever factored is k x k. If this matches the full method, none of the
    machinery is earning its cost.
    """
    n, p = Z.shape
    Zty = Z.T @ y
    scale = np.linalg.norm(Z, axis=0)
    support = np.argsort(-np.abs(Zty) / np.maximum(scale, 1e-12))[:k]
    Sigma_X_hat = (Z.T @ Z) / n - Sigma_a
    return refit_on_support_raw(Sigma_X_hat, Zty, n, support, psd_floor=psd_floor)


# ----------------------------------------------------------------------
# Corrected cross-validation for lambda (Datta & Zhang, Sec. 6.2)
# ----------------------------------------------------------------------

def corrected_cv_loss(Z_val: np.ndarray, y_val: np.ndarray,
                      Sigma_a: np.ndarray, beta: np.ndarray) -> float:
    """Bias-corrected held-out loss.

    The held-out design is corrupted too, so the naive validation loss is
    biased upward. With y = X beta* + eps and Z = X + A,

        E ||y_val - Z_val beta||^2 / n
            = (beta* - beta)' Sigma_X (beta* - beta) + sigma_e^2
              + beta' Sigma_a beta,

    and only the last term depends on the fit through the corruption. Removing
    it leaves an unbiased ranking of candidate betas -- and note it is the same
    correction as in the objective itself: subtracting beta' Sigma_a beta
    replaces Z_val'Z_val/n by its bias-corrected counterpart.
    """
    r = y_val - Z_val @ beta
    return float(r @ r) / len(y_val) - float(beta @ Sigma_a @ beta)


def lambda_grid(Z: np.ndarray, y: np.ndarray, n_lambdas: int = 20,
                ratio: float = 1e-2) -> np.ndarray:
    """Geometric grid down from the smallest lambda that zeroes beta.

    The regression is  1/2 beta' Sigma beta - <Z'y/n, beta> + lambda ||beta||_1,
    whose subgradient at 0 gives beta = 0 exactly when lambda >= ||Z'y/n||_inf,
    so that is the top of the grid.
    """
    lam_max = float(np.abs(Z.T @ y).max() / len(y))
    lam_max = max(lam_max, 1e-12)
    return np.geomspace(lam_max * ratio, lam_max, n_lambdas)[::-1]


def kfold_indices(n: int, folds: int, rng: np.random.Generator) -> list:
    """Shuffled (train, validation) index pairs."""
    if not 2 <= folds <= n:
        raise ValueError(f"folds must be in [2, n]; got {folds} with n={n}")
    order = rng.permutation(n)
    return [(np.setdiff1d(order, part, assume_unique=True), part)
            for part in np.array_split(order, folds)]


def select_lambda_cv(Z: np.ndarray, y: np.ndarray, Sigma_a: np.ndarray,
                     fit, lambdas=None, folds: int = 5, seed: int = 0,
                     n_lambdas: int = 20, ratio: float = 1e-2) -> tuple:
    """Pick lambda by k-fold corrected cross-validation.

    `fit(Z_train, y_train, Sigma_a, lam) -> beta` is called folds * len(lambdas)
    times, so the cost is that multiple of a single fit -- with the reweighted
    algorithm and a 20-point grid that is 100 fits per replication. Shrink
    n_lambdas before folds if it is too slow; the grid matters less than the
    correction does.

    Returns (best_lambda, losses) with losses the mean corrected loss per
    lambda, in the same order as the grid.
    """
    if lambdas is None:
        lambdas = lambda_grid(Z, y, n_lambdas=n_lambdas, ratio=ratio)
    lambdas = np.asarray(lambdas, dtype=float)

    splits = kfold_indices(len(y), folds, np.random.default_rng(seed))
    losses = np.zeros(len(lambdas))
    for j, lam in enumerate(lambdas):
        total = 0.0
        for train, val in splits:
            beta = fit(Z[train], y[train], Sigma_a, float(lam))
            total += corrected_cv_loss(Z[val], y[val], Sigma_a, beta)
        losses[j] = total / folds

    return float(lambdas[int(np.argmin(losses))]), losses


def default_lambda(n: int, p: int) -> float:
    """Universal-threshold scaling sqrt(2 log p / n) for the ell-1 penalty.

    This is the standard rate, but it carries no estimate of the noise scale:
    the optimal lambda is proportional to the residual standard deviation, so
    on a design with large sigma this default is too small by roughly that
    factor. Treat it as a starting point and tune -- see experiments/.
    """
    return float(np.sqrt(2.0 * np.log(max(p, 2)) / n))


def cocolasso(Z: np.ndarray, y: np.ndarray, Sigma_a: np.ndarray,
              lam: float | None = None, projection: str = "max",
              regression_solver=None, admm_rho: float = 1.0,
              admm_max_iter: int = 300, psd_floor: float = 1e-6) -> np.ndarray:
    """Plain CoCoLasso: one PSD projection of Sigma-hat_X, one regression solve.

    This is Algorithm 1 stripped of the GSM reweighting -- no k, no gamma
    schedule, no rho -- and is the baseline the reweighted version has to beat.
    Taking C = I in surrogate_design reproduces the Datta & Zhang construction
    verbatim: Z_tilde / sqrt(n) is the Cholesky factor of Sigma-tilde, and
    Z_tilde^T y_tilde = Z^T y.

    Note that reweighted_cocolasso with rho = 1 is mathematically identical to
    this, since C is then a multiple of the identity and the max-norm
    projection is positively homogeneous, so C cancels out of Sigma_t.

    R(beta) = lam * ||beta||_1 by default, with lam from default_lambda when not
    given. Pass regression_solver to override R entirely (solve_unpenalized for
    R = 0, make_refit_solver for lasso followed by a debiasing refit).
    """
    n, p = Z.shape
    Sigma_X_hat = (Z.T @ Z) / n - Sigma_a
    Zty = Z.T @ y

    floor = psd_floor * max(abs(np.trace(Sigma_X_hat)) / p, 1e-12)
    Sigma_tilde = project_corrected(Sigma_X_hat, psd_floor=floor,
                                    projection=projection, admm_rho=admm_rho,
                                    admm_max_iter=admm_max_iter)

    Z_tilde, y_tilde = surrogate_design(Sigma_tilde, np.ones(p), Zty, n)

    if regression_solver is None:
        if lam is None:
            lam = default_lambda(n, p)
        regression_solver = (lambda Zt, yt, bp:
                             solve_lasso(Zt, yt, n, lam, beta_init=bp))
    return regression_solver(Z_tilde, y_tilde, np.zeros(p))


# ----------------------------------------------------------------------
# Convenience: geometric homotopy schedule
# ----------------------------------------------------------------------

def geometric_schedule(gamma1: float, T: int, growth: float = 2.0,
                        gamma_T_inf: bool = False) -> list:
    """Build 0 = gamma_0 < gamma_1 < ... < gamma_T (<= inf), as recommended in
    the paper's 'Homotopy schedule' remark: a geometric grid after a small gamma_1."""
    schedule = [0.0]
    g = gamma1
    for t in range(T):
        if t == T - 1 and gamma_T_inf:
            schedule.append(np.inf)
        else:
            schedule.append(g)
            g *= growth
    return schedule


# ----------------------------------------------------------------------
# Section 4: Algorithm 1
# ----------------------------------------------------------------------

def reweighted_cocolasso(Z: np.ndarray, y: np.ndarray, Sigma_a: np.ndarray, k: int,
                          gamma_schedule: list | None = None,
                          gamma_rule: str = "geometric",
                          gamma1: float = 0.5, growth: float = 2.0,
                          f_tol: float | None = 1e-8, min_iter: int = 5,
                          max_iter: int = 100,
                          rho: float = 0.1, lam: float | None = None,
                          projection: str = "max",
                          regression_solver=None, admm_rho: float = 1.0,
                          admm_max_iter: int = 300, weight_clip: float = 1e-6,
                          psd_floor: float = 1e-6, return_info: bool = False,
                          verbose: bool = False):
    """Algorithm 1: reweighted covariance-corrected sparse regression.

    Parameters
    ----------
    Z, y        : noisy design (n x p) and response (n,)
    Sigma_a     : known measurement-error covariance (p x p)
    k           : target sparsity level
    gamma_schedule : increasing sequence [0=gamma_0, gamma_1, ..., gamma_T],
                      gamma_T may be np.inf. Pass None to generate the geometric
                      grid gamma_t = gamma1 * growth^(t-1) on the fly and let
                      f_tol decide when to stop instead of a preset T.
    gamma1, growth : parameters of that generated grid (ignored when an explicit
                      gamma_schedule is given)
    f_tol       : stop once ||f_{t+1} - f_t||_inf < f_tol, i.e. once the weights
                  have converged. The homotopy reaches a fixed point well before
                  a long schedule runs out -- once the GSM weights saturate at
                  {0, 1} the damped weights stop moving and every further
                  iteration reproduces the same Sigma_t -- so a preset T either
                  stops early or burns iterations. Pass None to disable and run
                  the schedule to its end.
    min_iter    : iterations to run before f_tol may trigger. At small gamma_1
                  the first weights are still near uniform and can move by less
                  than f_tol, which would stop the path before it has started.
    max_iter    : cap on iterations when the schedule is generated
    rho         : mixing parameter in [0, 1], eq. (8)
    lam         : weight of the default penalty R(beta) = lam * ||beta||_1.
                  Defaults to default_lambda(n, p); ignored if regression_solver
                  is given. Use the same lam here and in cocolasso() when
                  comparing the two -- the comparison is meaningless otherwise.
    regression_solver : optional callable(Z_tilde, y_tilde, beta_prev) -> beta_t,
                  overriding R entirely. It receives the surrogate data of
                  surrogate_design and should minimise
                  1/(2n) ||y_tilde - Z_tilde beta||^2 + R(beta).
                  solve_unpenalized gives R = 0; make_refit_solver gives lasso
                  followed by a debiasing refit.
    weight_clip : keeps f_t(i) <= 1 - weight_clip so C_t stays invertible (Remark 2)
    psd_floor   : projects onto {B >= psd_floor * tr(M)/p * I} rather than onto
                  the PSD cone, so that B_tilde admits a Cholesky factor; the
                  max-norm projection is singular at psd_floor = 0

    Returns
    -------
    beta_T : the coefficient estimate at the end of the path
    (beta_T, info) if return_info, with info = {"iters", "gamma", "f_step",
    "converged"}
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0, 1]")

    if gamma_rule not in ("geometric", "amir"):
        raise ValueError("gamma_rule must be 'geometric' or 'amir'")

    if gamma_rule == "amir":
        if gamma_schedule is not None:
            raise ValueError("gamma_rule='amir' generates its own schedule; "
                             "leave gamma_schedule as None")
        T, gamma_at = max_iter, None
    elif gamma_schedule is None:
        T = max_iter
        def gamma_at(t):                      # gamma_1, gamma_2, ... generated
            return gamma1 * growth ** t
    else:
        gamma_schedule = list(gamma_schedule)
        if gamma_schedule[0] != 0:
            raise ValueError("gamma_schedule must start at gamma_0 = 0")
        T = len(gamma_schedule) - 1
        if T < 1:
            raise ValueError("gamma_schedule needs at least [0, gamma_1]")
        def gamma_at(t):
            return gamma_schedule[t + 1]

    n, p = Z.shape
    Sigma_Z_hat = (Z.T @ Z) / n                 # Sigma-hat_Z
    Sigma_X_hat = Sigma_Z_hat - Sigma_a          # Sigma-hat_X, may be indefinite
    Zty = Z.T @ y

    if regression_solver is None:
        if lam is None:
            lam = default_lambda(n, p)
        regression_solver = (lambda Zt, yt, bp:
                             solve_lasso(Zt, yt, n, lam, beta_init=bp))

    m = p - k
    f = np.full(p, rho / p + (1.0 - rho) * m / p)  # f_0, eq. after Step 3

    beta = np.zeros(p)
    f_step, converged, t = np.inf, False, 0
    gamma_next, n_sparse_w = 0.0, 0
    for t in range(T):
        beta_prev, f_prev = beta, f

        f_clipped = np.clip(f, 0.0, 1.0 - weight_clip)
        c = 1.0 - f_clipped                       # diagonal of C_t
        c_sqrt = np.sqrt(c)
        weighted = np.outer(c_sqrt, c_sqrt) * Sigma_X_hat

        # Steps 6-7: project, then read off the Cholesky factor of Sigma_t
        # without ever forming Sigma_t = C^-1/2 B_tilde C^-1/2.
        floor = psd_floor * max(abs(np.trace(weighted)) / p, 1e-12)
        B_tilde = project_corrected(weighted, psd_floor=floor,
                                    projection=projection, admm_rho=admm_rho,
                                    admm_max_iter=admm_max_iter)

        Z_tilde, y_tilde = surrogate_design(B_tilde, c, Zty, n)

        beta = regression_solver(Z_tilde, y_tilde, beta)

        # Choose the gamma at which the next weights are built. The 'amir' rule
        # needs the current beta, so it is decided here rather than up front.
        if gamma_rule == "amir":
            if gamma_next == np.inf:
                pass                                    # already at the last step
            elif t == 0:
                md = max_diff_of_sums(beta, k)          # gamma_1 from the spread
                gamma_next = GAMMA_FIRST_MAX_DIFF / md if md > 0 else 1.0
            else:
                if weight_nonsparsity(gsm_weights(beta, k, gamma_next),
                                      k) <= W_SPARSITY_THRESH:
                    n_sparse_w += 1
                else:
                    n_sparse_w = 0
                gamma_next = (np.inf if n_sparse_w >= N_SPARSE_W_TO_STOP
                              else next_gamma_amir(gamma_next, beta, k))
        else:
            gamma_next = gamma_at(t)

        f = damped_weights(beta, k, gamma_next, rho)
        f_step = np.abs(f - f_prev).max()

        if verbose:
            w = gsm_weights(beta, k, gamma_next)
            print(f"t={t:3d}  gamma={gamma_next:>11.4g}  nnz={np.sum(np.abs(beta) > 1e-6):>4d}"
                  f"  d_beta={np.linalg.norm(beta - beta_prev):>9.2e}"
                  f"  d_f={f_step:>9.2e}  w:[{w.min():.4f},{w.max():.4f}]"
                  f"  f:[{f.min():.4f},{f.max():.4f}]  c_min={1-f.max():.4g}",
                  flush=True)

        # Stop once the weights stop moving: the GSM weights saturate at {0, 1}
        # long before a long schedule ends, after which f is fixed and every
        # further iteration rebuilds the identical Sigma_t.
        if f_tol is not None and t + 1 >= min_iter and f_step < f_tol:
            converged = True
            if verbose:
                print(f"    converged: ||f_t - f_(t-1)||_inf = {f_step:.3e} < {f_tol:g} "
                      f"after {t + 1} iterations", flush=True)
            break

    if return_info:
        return beta, {"iters": t + 1, "gamma": gamma_next,
                      "f_step": f_step, "converged": converged}
    return beta


# ----------------------------------------------------------------------
# Demo / sanity check
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n, p, k = 300, 20, 4
    beta_star = np.zeros(p)
    support = rng.choice(p, size=k, replace=False)
    beta_star[support] = rng.choice([-1, 1], size=k) * rng.uniform(1.5, 3.0, size=k)

    X = rng.standard_normal((n, p))
    sigma_a2 = 0.2
    Sigma_a = sigma_a2 * np.eye(p)
    A = rng.normal(scale=np.sqrt(sigma_a2), size=(n, p))
    Z = X + A
    y = X @ beta_star + rng.normal(scale=0.5, size=n)

    schedule = geometric_schedule(gamma1=0.5, T=20, growth=2.0, gamma_T_inf=True)

    lam = 0.15
    beta_hat = reweighted_cocolasso(
        Z, y, Sigma_a, k=k, gamma_schedule=schedule,
        rho=0.1, lam=lam, verbose=True,
    )
    # Same lam for the baseline, or the comparison says nothing.
    beta_base = cocolasso(Z, y, Sigma_a, lam=lam)

    print("\ntrue support   :", sorted(support.tolist()))
    print("recovered top-k:", sorted(np.argsort(-np.abs(beta_hat))[:k].tolist()))
    print(f"\nl2 error   reweighted: {np.linalg.norm(beta_hat - beta_star):.4f}"
          f"   CoCoLasso: {np.linalg.norm(beta_base - beta_star):.4f}")
    print("\nbeta*   :", np.round(beta_star, 3))
    print("beta_hat:", np.round(beta_hat, 3))
