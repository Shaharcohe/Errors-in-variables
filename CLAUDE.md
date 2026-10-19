# CLAUDE.md

## User preferences

- **Keep answers short.** The user always prefers brief, to-the-point replies in Claude Code.

## What this is

Research code for errors-in-variables sparse regression: CoCoLasso (Datta & Zhang 2017)
plus a reweighted homotopy using the GSM penalty (Amir, Basri & Nadler 2021).
`README.md` has the full math and design notes; read it before changing algorithms.

## Layout

| file | role |
|---|---|
| `eiv_algorithm.py` | all the math: projections (max-norm ADMM / Frobenius), surrogate design, lasso/refit solvers, `cocolasso`, `reweighted_cocolasso`, `two_step_cocolasso`, `naive_topk`, corrected CV |
| `config.py` | data generators (`ar`, `cs`, `identity`), `beta*` presets, `snr()` |
| `test.py` | `run_test(...)`: one configuration → metrics (MSE, PE, top-k, time) |
| `main.py` | CLI; comma-separated values sweep; writes JSON to `results/` (or `--out`) |
| `run_lsf.sh` | LSF job wrapper around `main.py` |
| `submit_sweep.sh` | submits the p=500 sweep (`--dry-run`, `--plain`/`--refit`/`--all`) to `results/sweep_p500/<setting>/<arm>.json` |
| `export_runs_csv.py` | sweep JSONs → `results/sweep_p500_runs.csv` (one row per repetition) |
| `paired_diff.py`, `plot_settings.py`, `plot_all_losses.py` | paired/per-setting plots from that CSV |
| `plot_cocolasso_vs_reweighted.py`, `plot_median_pe.py` | older summary plots |
| `plot_covariance_error.py` | tracks `‖Sigma_t − Sigma_X‖` along the homotopy; caches in `results/covariance_error/` |
| `track_iterations.py` | per-iteration metrics (cov error by S/Sᶜ block, β error, top-k overlap) at sweep settings → `results/iteration_tracking/<setting>/<arm>.json` + `.npz` of β_t |
| `submit_tracking.sh` | LSF submit for `track_iterations.py` (24 settings × 4 arms; `--dry-run`, `--smoke`) |
| `plot_iterations.py` | per-setting `iterations.png` / `iterations_delta.png` and `overview_<metric>.png` |
| `submit_twostep.sh` | LSF submit of `twostep` (24 settings × max/frobenius, `--gamma=`, `--dry-run`) → `results/sweep_p500/<setting>/twostep_g<γ>_<norm>.json` |
| `topk_overlap.py`, `plot_topk_overlap.py` | top-10 overlaps with true top-10 / top-3 (from tracked β's + sweep CSV) → `results/topk_overlap_*` |

Algorithms: `cocolasso`, `cocolasso_refit`, `reweighted`, `reweighted_refit`, `twostep`,
`twostep_refit`, `naive`. `twostep` = CoCoLasso (C = I), then one reweighting at a fixed
`--gamma` (default 1) with ρ = 0 (no `rho` parameter).
Norms: `max`, `frobenius`, `max_then_frobenius` (max on first homotopy iteration only).

## Commands

```bash
python config.py                     # sanity: SNR must be 2.3611 (AR) and 3.1944 (CS)
python eiv_algorithm.py              # built-in demo
python main.py --p 60 --n_reps 5 --algorithm cocolasso,reweighted   # quick end-to-end check
./submit_sweep.sh --dry-run
python export_runs_csv.py && python plot_settings.py
```

No test suite; the checks above are the verification. numpy, scipy, matplotlib and
`sparse_approx_gsm` live in `~/.local/lib/python3.9`. On the login node use `python`
(`/usr/bin/python`); **compute nodes have no `/usr/bin/python`** — LSF jobs must
`module load Python/3.9.6-GCCcore-11.2.0-bare` and call `python3` (see `submit_tracking.sh`).

## Gotchas

- Reweighting acts **only through the PSD projection**: if `Sigma_X_hat` is already PSD,
  reweighted == CoCoLasso. `rho=1` must reproduce plain CoCoLasso exactly (good regression check).
- Max-norm ADMM is **not scale invariant** in practice: projecting c·Σ̂ ≠ c·proj(Σ̂). So the
  homotopy's t = 0 (C = cI) differs from CoCoLasso by ~8% under max (exact under Frobenius).
- `psd_floor` is required (max-norm projection lands on the PSD boundary → Cholesky fails).
- `k` is a target for `reweighted_cocolasso` but only a cap for `make_refit_solver`.
  `true_k` is for scoring only and never reaches an algorithm.
- Runs are paired by construction: data is deterministic in
  `(seed, model, n, p, sigma_a, sigma_e, beta*)`; repetition i uses `seed + i`.
- `main.py` writes JSON only after the last repetition. `reweighted`+`max` is slow
  (minutes/fit at p=500), so the sweep chunks it into 4×25 seeds under `<setting>/parts/`;
  `export_runs_csv.py` merges parts automatically.
- `run_lsf.sh` / `submit_sweep.sh` hardcode `REPO_DIR` and `PYTHON` under
  `/home/projects/nyosef/...` (inaccessible), not this checkout — fix before submitting from here.
- `results/` and `experiments/` are gitignored. README references `experiments/` and
  `submit_sweep.sh` references `merge_chunks.py`; neither exists in this checkout.
- Naming: `sigma_a` = paper's tau, `sigma_e` = paper's sigma.
