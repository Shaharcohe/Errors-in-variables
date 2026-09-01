#!/bin/bash
# Submit the p=500 sweep: 24 settings x 5 algorithm/norm arms.
#
#   ./submit_sweep.sh --dry-run    print the bsub lines, submit nothing
#   ./submit_sweep.sh              submit
#
# Design (see .claude/plans/ for the reasoning):
#
#   model    ar, cs
#   sigma_a  0.75, 1.0, 1.25
#   beta     datta_zhang, and weak_sparse at alpha 1.0 / 1.5 / 2.0
#   arms     cocolasso and reweighted, under max / frobenius /
#            max_then_frobenius (the last only differs for reweighted)
#
# Everything else is fixed: n=100 p=500 k=10 true_k=3 sigma_e=3 rho=0.5 lam=1,
# 100 repetitions at seeds 0..99. Seeds and data generation are deterministic
# given (seed, model, n, p, sigma_a, sigma_e, beta*), so every arm within a
# setting sees bit-identical data and the comparison is paired by construction.
#
# One invocation per (setting, arm): main.py writes a single JSON per run
# holding all swept combos in one "runs" list, so the requested one-file-per-
# algorithm layout needs separate invocations with --out rather than one big
# comma-separated sweep.
#
# reweighted+max is the expensive arm (~350-425 s/fit at p=500, i.e. ~10 h for
# 100 reps). These queues are PREEMPTABLE and main.py only writes its JSON
# after the last repetition, so a preemption at hour 9 loses everything. That
# arm is therefore split into 4 chunks of 25 reps (~2.5 h each) via --seed,
# which preserves the global seed set 0..99 exactly; merge_chunks.py stitches
# them back into one file per setting afterwards.

set -euo pipefail

REPO_DIR="/home/projects/nyosef/shachaco/Errors-in-variables"
OUT_ROOT="results/sweep_p500"
QUEUE="medium"          # 72 h limit; all 59 prior jobs used it, and `short` is
                        # currently backlogged ~2100 pending vs ~330 running
MEM=1000                # observed peak across every prior job was 158 MB
# Deliberately generous. A smoke test at p=60 took ~26 s/fit for the hybrid,
# which means the FISTA lasso (20000 iters, tol 1e-10) dominates, not the
# projection -- so per-fit cost scales worse than the O(p^3) eigendecomposition
# model suggested. main.py writes its JSON only after the final repetition, so
# a TERM_RUNLIMIT kill loses the entire job; over-requesting walltime costs
# only some backfill priority. medium allows 72 h.
WALL_FAST="24:00"
WALL_SLOW="48:00"       # per 25-rep chunk of reweighted+max

DRY_RUN=0
ARM_SET="plain"         # plain | refit | all
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --plain)   ARM_SET="plain" ;;
        --refit)   ARM_SET="refit" ;;
        --all)     ARM_SET="all" ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$REPO_DIR"
mkdir -p logs "$OUT_ROOT"

MODELS=(ar cs)
SIGMAS=(0.75 1.0 1.25)
CHUNK_SEEDS=(0 25 50 75)

# arm_name:algorithm:norm
#
# The refit arms are the same five configurations with a debiasing refit on the
# top-k coordinates the lasso selected: make_refit_solver caps the support at k
# (=10 here, matching what the reweighted homotopy is handed) and refits from
# the raw bias-corrected covariance block with a small ridge.
#
# Selected with --plain / --refit / --all rather than always submitting both:
# the skip-if-exists guard below only sees finished jobs, so re-running the
# script while the plain sweep is still in flight would resubmit every one of
# them as a duplicate.
PLAIN_ARMS=(
    "cocolasso_max:cocolasso:max"
    "cocolasso_frobenius:cocolasso:frobenius"
    "reweighted_max:reweighted:max"
    "reweighted_frobenius:reweighted:frobenius"
    "reweighted_max_then_frobenius:reweighted:max_then_frobenius"
)
REFIT_ARMS=(
    "cocolasso_refit_max:cocolasso_refit:max"
    "cocolasso_refit_frobenius:cocolasso_refit:frobenius"
    "reweighted_refit_max:reweighted_refit:max"
    "reweighted_refit_frobenius:reweighted_refit:frobenius"
    "reweighted_refit_max_then_frobenius:reweighted_refit:max_then_frobenius"
)

case "$ARM_SET" in
    plain) ARMS=("${PLAIN_ARMS[@]}") ;;
    refit) ARMS=("${REFIT_ARMS[@]}") ;;
    all)   ARMS=("${PLAIN_ARMS[@]}" "${REFIT_ARMS[@]}") ;;
esac

# beta_tag:preset:alpha_flag  -- datta_zhang carries NO --alpha. alpha is an
# unconditional sweep axis in main.py, so passing it to a preset that ignores
# it would produce bit-identical duplicate runs.
BETAS=(
    "datta_zhang:datta_zhang:"
    "weak_sparse_a1.0:weak_sparse:--alpha 1.0"
    "weak_sparse_a1.5:weak_sparse:--alpha 1.5"
    "weak_sparse_a2.0:weak_sparse:--alpha 2.0"
)

n_submit=0
n_skip=0

submit() {          # $1 job name, $2 walltime, $3 out json, rest: main.py args
    local jobname="$1" wall="$2" outfile="$3"
    shift 3
    if [[ -e "$outfile" ]]; then
        n_skip=$((n_skip + 1))
        return
    fi
    n_submit=$((n_submit + 1))
    # -oo/-eo, not -o/-e: LSF appends on re-dispatch, and a preempted job would
    # otherwise interleave two runs into one log.
    local cmd=(bsub -q "$QUEUE" -n 1 -R "rusage[mem=$MEM]" -W "$wall"
               -J "$jobname" -oo "logs/%J.out" -eo "logs/%J.err"
               ./run_lsf.sh "$@" --out "$outfile")
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
    else
        "${cmd[@]}"
    fi
}

for model in "${MODELS[@]}"; do
  for sigma in "${SIGMAS[@]}"; do
    for beta_spec in "${BETAS[@]}"; do
      IFS=: read -r beta_tag preset alpha_flag <<< "$beta_spec"
      setting="${model}_sigma${sigma}_${beta_tag}"
      setting_dir="$OUT_ROOT/$setting"

      for arm_spec in "${ARMS[@]}"; do
        IFS=: read -r arm algorithm norm <<< "$arm_spec"

        # Common arguments. $alpha_flag is intentionally unquoted so that an
        # empty value expands to nothing rather than to an empty argument.
        common=(--model "$model" --n 100 --p 500 --sigma_a "$sigma" --sigma_e 3.0
                --beta "$preset" $alpha_flag --lam 1.0
                --algorithm "$algorithm" --norm "$norm"
                --k 10 --true_k 3 --rho 0.5 --quiet --progress)

        # Chunk the expensive arm: a full-max homotopy runs ~185 s/fit at
        # p=500, so 100 reps is ~5 h in one preemptable job that saves nothing
        # until its last repetition. Applies to reweighted and reweighted_refit
        # alike; every other arm is well under an hour.
        if [[ "$algorithm" == reweighted* && "$norm" == "max" ]]; then
            for s in "${CHUNK_SEEDS[@]}"; do
                submit "eiv-${setting}-${arm}-s${s}" "$WALL_SLOW" \
                       "$setting_dir/parts/${arm}.seed${s}.json" \
                       "${common[@]}" --n_reps 25 --seed "$s"
            done
        else
            submit "eiv-${setting}-${arm}" "$WALL_FAST" \
                   "$setting_dir/${arm}.json" \
                   "${common[@]}" --n_reps 100 --seed 0
        fi
      done
    done
  done
done

echo
echo "submitted: $n_submit   skipped (output already exists): $n_skip"
[[ $DRY_RUN -eq 1 ]] && echo "(dry run -- nothing was actually submitted)"
exit 0
