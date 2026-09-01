# Errors-in-Variables Sparse Regression

Reweighted covariance-corrected sparse regression: the CoCoLasso projection of
Datta & Zhang (2017) combined with the generalized soft-min (GSM) penalty of
Amir, Basri & Nadler (2021).

## The problem

We observe a response `y` in `R^n` and a *corrupted* design

```
Z = X + A,        rows of A i.i.d. mean zero with known covariance Sigma_a
y = X b* + eps,   b* is k-sparse
```

`X` is never seen. The naive Gram matrix `Z'Z/n` is biased upward by `Sigma_a`,
so the bias-corrected `Sigma_X_hat = Z'Z/n - Sigma_a` is the right object — but
it need not be positive semidefinite, which would make the regression
non-convex. CoCoLasso fixes that by projecting it onto the PSD cone. This code
adds a homotopy that reweights the coordinates through the GSM surrogate for the
trimmed lasso.

## Layout

```
eiv_algorithm.py   the algorithms
config.py          data generation
test.py            run one configuration, return metrics
main.py            terminal front end, saves JSON
results/           output, one file per invocation
experiments/       the exploratory scripts and their raw output (see its README)
```

`config`, `test` and `main` are thin: all the mathematics lives in
`eiv_algorithm.py` and the other three only wire it up.

---

## eiv_algorithm.py

### The pipeline

Both estimators run the same three steps. Given weights `f` in `[0,1]^p` and
`C = diag(1 - f)`:

1. **Project.** `B = argmin over {B >= 0} of ||C^1/2 Sigma_X_hat C^1/2 - B||`,
   in the max norm (Datta & Zhang) or the Frobenius norm.
2. **Factor.** Read the Cholesky factor of `Sigma_t = C^-1/2 B C^-1/2` off `B`
   directly, and build surrogate data `(Zt, yt)` with

   ```
   Zt' Zt / n = Sigma_t        Zt' yt = Z' y
   ```

   so the quadratic program becomes an ordinary least-squares fit that any
   regression solver can take.
3. **Regress.** `min over beta of  1/(2n) ||yt - Zt beta||^2 + R(beta)`.

`cocolasso()` does this once with `C = I`. `reweighted_cocolasso()` iterates it,
rebuilding `f` from the current `beta` at a growing inverse temperature `gamma`.

### Key functions

| function | what it does |
|---|---|
| `cocolasso(Z, y, Sigma_a, lam, projection, ...)` | the published baseline: one projection, one solve. Takes no `k`. |
| `reweighted_cocolasso(Z, y, Sigma_a, k, rho, lam, ...)` | the homotopy. `rho` damps the weights, `gamma_rule` picks the schedule. |
| `naive_topk(Z, y, Sigma_a, k)` | marginal screening, then an unpenalised fit on the survivors. No projection of the full matrix. |
| `project_corrected(M, psd_floor, projection)` | dispatches to `cocolasso_projection` (max norm, ADMM) or `frobenius_projection` (closed form). |
| `surrogate_design(B_tilde, c, Zty, n)` | step 2 above; returns `(Zt, yt)`. |
| `solve_lasso`, `solve_unpenalized`, `make_refit_solver` | the `R(beta)` options, all matching one `regression_solver` protocol. |
| `select_lambda_cv(Z, y, Sigma_a, fit, ...)` | k-fold cross-validation with the corrected loss. |

### Things worth knowing before changing anything

**`psd_floor` is not optional.** The max-norm projection lands on the *boundary*
of the PSD cone whenever `Sigma_X_hat` is indefinite: the minimiser cannot be
interior, since from any interior point you can move towards `Sigma_X_hat`, stay
feasible, and strictly decrease the distance. So the projection is singular and
`np.linalg.cholesky` raises. Projecting onto `{B >= eps*I}` instead keeps the
guarantee `||B - Sigma_X||_max <= 2 ||Sigma_X_hat - Sigma_X||_max` — which only
needs `Sigma_X` itself to be feasible — while making the factorisation possible.
With `R = 0` it is also the *only* regularisation left, so it behaves as a ridge
parameter and should be swept like one.

**`rho` is structural, not cosmetic.** The damped weights satisfy
`f_i` in `[rho/p, 1 - rho(p-1)/p]`, so `rho > 0` is what keeps `C` invertible —
it plays the role of the clipping level in Remark 2 of the write-up. It also
bounds how much `C` can distinguish coordinates: the contrast ratio is
`(p - rho) / (rho (p-1))`, which is 2:1 at `rho = 0.5` and 10:1 at `rho = 0.1`.
At `rho = 1` the algorithm reduces *exactly* to plain CoCoLasso, because `C`
becomes a multiple of the identity and the max-norm projection is positively
homogeneous, so `C` cancels out of `Sigma_t`. That makes `rho=1` a free
correctness check on any future change.

**`k` means two different things.** For `reweighted_cocolasso` it is a target:
the GSM weights try to hold exactly `k` coordinates active. For
`make_refit_solver` it is only a *cap* on the lasso's own nonzeros, so a `k`
above the selected support size does nothing at all. A refit handed the true `k`
looks far stronger than one handed a loose upper bound.

**`projection="frobenius"` is one eigendecomposition** against the 50–300 ADMM
iterations the max norm needs. The theory prefers the max norm: the same
argument in Frobenius norm controls the Frobenius error, and converting back to
the max norm costs a factor of `p`. Whether that matters in practice is an
empirical question, and `experiments/check_frobenius.py` measures it directly.

**The gamma schedule.** `gamma_rule="geometric"` multiplies by `growth` each
step. `gamma_rule="amir"` ports the reference GSM implementation: `gamma_1` is
set relative to the spread of `|beta|` rather than as an absolute constant, the
growth factor is 1.02, and gamma keeps being multiplied until the weights
actually move by a set amount — so the step size is governed by weight movement,
not by gamma. The homotopy stops when the weights stop moving (`f_tol`), which
typically happens well before a long fixed schedule would end.

---

## config.py

Every generator has the same signature and returns the same thing:

```python
Z, y = config.ar(n, p, sigma_a, sigma_e, beta_star, rng)
```

with `X ~ N(0, Sigma_x)`, `A ~ N(0, sigma_a^2 I)`, `eps ~ N(0, sigma_e^2 I)`,
all mean zero. `Sigma_x` is what distinguishes the models, so each has a
matching covariance function taking `p` alone, and `MODELS` ties them together:

| name | `Sigma_x` | note |
|---|---|---|
| `ar` | `0.5^abs(i-j)` | autoregressive |
| `cs` | `0.5 + 0.5*I(i=j)` | compound symmetry; nearly rank one for large `p`, and far harder for support recovery than AR at the same nominal correlation |
| `identity` | `I` | uncorrelated |

`beta*` presets live here too (`datta_zhang`, `two_tier`, `equal`,
`weak_sparse`) via `make_beta(preset, p)`. The first three are hard sparse
(exact zeros outside a fixed support); `weak_sparse` decays as a power law
instead, so it has no exact zeros and is only ever approximately k-sparse.
`snr(model, p, beta_star, sigma_e)` returns
`b*' Sigma_x b* / sigma_e^2`; it depends on `Sigma_x`, `beta*` and `sigma_e` at
once, which makes it the single best check that a design is coded correctly. The
Datta & Zhang additive-errors case must give **2.36** for AR and **3.20** for CS.

Naming: `sigma_a` is what the paper calls `tau`, `sigma_e` is what it calls
`sigma`.

---

## test.py

```python
run_test(model, n, p, sigma_a, sigma_e, beta_star, lam, seeds, n_reps,
         algorithm, norm, k, true_k, algo_params, cv) -> dict
```

Five algorithms: `cocolasso`, `cocolasso_refit`, `reweighted`,
`reweighted_refit`, `naive`. `norm` is `"max"` or `"frobenius"`.

### Metrics

| metric | definition |
|---|---|
| MSE | `\|\|beta_hat - beta*\|\|^2` |
| PE | `(beta* - beta_hat)' Sigma_x (beta* - beta_hat)`, the paper's definition. The *population* prediction error, exact given `Sigma_x`, so unlike a held-out test sample it adds no Monte-Carlo noise of its own; measured on the true covariates, not on `Z`. |
| top-k | 1 if the `true_k` largest-magnitude entries of `beta_hat` are exactly the support of `beta*` |
| time | seconds per fit, timed around the fit alone so that data generation — common to every algorithm — does not flatter the slow ones |

`true_k` is used only for scoring and never reaches an algorithm; `k` is what
the algorithms receive.

### Cross-validated lambda

Passing `lam="cv"` selects it per replication by k-fold cross-validation with
the **corrected** loss. The held-out design is corrupted too, so the naive
validation loss is biased: with `y = X b* + eps` and `Z = X + A`,

```
E ||y_val - Z_val b||^2 / n
    = (b* - b)' Sigma_X (b* - b) + sigma_e^2 + b' Sigma_a b
```

and only the last term depends on the fit through the corruption. Subtracting it
is exactly the same correction as in the objective itself — it replaces
`Z_val' Z_val / n` by its bias-corrected counterpart.

Selection is redone on every replication because it is part of the estimator;
reusing one lambda across replications would understate its variance. The cost
is `folds * n_lambdas` extra fits per replication, so shrink `cv_lambdas` before
`cv_folds` if it is too slow.

### What the result records

Alongside the metrics, `run_test` records the parameters each algorithm
*actually ran with*, including the defaults it never saw, read off the
signatures of the functions it calls — a result saying `algo_params={}` would
not say which `psd_floor` produced it. It also records which arguments the
algorithm **ignored** (`naive` uses neither `lam` nor `norm`; `cocolasso`
ignores every reweighting parameter) rather than accepting them silently, and
both the requested lambda and the ones actually selected.

---

## main.py

```bash
python main.py                          # walk through the parameters, defaults in [..]
python main.py --model ar --n_reps 20   # run straight through
python main.py --interactive --p 800    # prompt, but start from p = 800
```

Sweepable arguments take comma-separated lists and main runs their product:

```bash
python main.py --lam 1.0,2.0 --algorithm cocolasso,reweighted --sigma_a 0.75,1.25
python main.py --lam cv --cv_lambdas 12 --algorithm reweighted
python main.py --beta weak_sparse --alpha 1.32   # sharper decay; alpha is ignored by the other presets
```

Validation is immediate: a value is checked the moment it is entered, so an
interactive session reports the problem at that prompt and asks again rather
than failing after all the questions have been asked. Cross-parameter checks
(`k` against `p`, `beta*` constructible at that `p`) run before any fit starts,
so a typo in a swept list fails at once rather than after the earlier
combinations have already run; interactively, those offer a correction too.

Warnings rather than errors for things that are legal but almost always
mistakes: `true_k` not matching the number of nonzeros in `beta*`, `k` below
`true_k`, `sigma_a=0`, `lam=0`, `n_reps=1`.

Output goes to `results/<model>_<algorithm>_<MMDD>-<HHMM>.json`, with a counter
appended rather than overwriting when two runs land in the same minute. The file
records every parameter as resolved — whether given or defaulted — so it alone
reproduces the run, plus per-replication values and the seeds. Runs that share
seeds are paired by construction and can be compared that way afterwards, even
though `test.py` computes no differences itself.

---

## Verification

```bash
python config.py
```

must print SNR **2.3611** for AR and **3.1944** for CS.

```bash
python eiv_algorithm.py
```

runs the built-in demo. Note that in its `n=300, p=20` regime `Sigma_X_hat` is
already PSD, so the projection is the identity map and `C` cancels exactly —
the reweighted and plain estimates therefore print as identical. That is correct
behaviour, not a bug, and it is the single most important thing to remember
about this method: **the reweighting can only act through the projection**, so
it does nothing at all when `Sigma_X_hat` is already PSD.

```bash
python main.py --p 60 --n_reps 5 --algorithm cocolasso,reweighted
```

is a quick end-to-end check.

## References

- A. Datta and C.-H. Zhang, *CoCoLasso for High-Dimensional Error-in-Variables Regression*, Ann. Statist. 45(6):2400–2426, 2017.
- T. Amir, R. Basri and B. Nadler, *The Trimmed Lasso: Sparse Recovery Guarantees and Practical Optimization by the Generalized Soft-Min Penalty*, SIAM J. Math. Data Sci. 3(3):900–929, 2021.
