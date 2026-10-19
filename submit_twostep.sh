#!/bin/bash
# Submit the two-step variant (eiv_algorithm.two_step_cocolasso) on the p=500
# sweep: 24 settings x {max, frobenius}.
#
#   ./submit_twostep.sh --dry-run          print the bsub lines, submit nothing
#   ./submit_twostep.sh                    submit, gamma = 1
#   ./submit_twostep.sh --gamma=4          another fixed gamma (its own arm name)
#
# Same settings, seeds (0..99) and fixed parameters as submit_sweep.sh, so the
# runs pair with every existing arm. No --rho: the variant always uses rho = 0.
# Output: results/sweep_p500/<setting>/twostep_g<gamma>_<norm>.json, which
# export_runs_csv.py picks up as arm twostep_g<gamma>_<norm>.
#
# Calls python directly rather than through run_lsf.sh, as submit_tracking.sh
# does: that wrapper hardcodes REPO_DIR and PYTHON under /home/projects/nyosef/.
#
# Two projections per fit, so max is ~2x CoCoLasso max (~35-50 s/fit, ~1-1.5 h
# per 100 reps) and needs no chunking; frobenius is ~1 s/fit.

set -euo pipefail

REPO_DIR="/home/projects/nadler/shachaco/Errors-in-variables"
PY_MODULE="Python/3.9.6-GCCcore-11.2.0-bare"
PYTHON="python3"                # compute nodes have no /usr/bin/python
OUT_ROOT="results/sweep_p500"
QUEUE="medium"
MEM=1000
WALL="12:00"

DRY_RUN=0
GAMMA="1"
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --gamma=*) GAMMA="${arg#--gamma=}" ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$REPO_DIR"
mkdir -p logs "$OUT_ROOT"

ENV_PREFIX="module load $PY_MODULE && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 &&"

n_submit=0
n_skip=0

submit() {          # $1 job name, $2 out json, rest: main.py args
    local jobname="$1" outfile="$2"
    shift 2
    if [[ -e "$outfile" ]]; then
        n_skip=$((n_skip + 1))
        return
    fi
    n_submit=$((n_submit + 1))
    local cmd=(bsub -q "$QUEUE" -n 1 -R "rusage[mem=$MEM]" -W "$WALL"
               -J "$jobname" -oo "logs/%J.out" -eo "logs/%J.err"
               "$ENV_PREFIX cd $REPO_DIR && $PYTHON main.py $* --out $outfile")
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
    else
        "${cmd[@]}"
    fi
}

MODELS=(ar cs)
SIGMAS=(0.75 1.0 1.25)
NORMS=(max frobenius)
# beta_tag:preset:alpha_flag -- datta_zhang carries no --alpha, as in submit_sweep.sh.
BETAS=(
    "datta_zhang:datta_zhang:"
    "weak_sparse_a1.0:weak_sparse:--alpha 1.0"
    "weak_sparse_a1.5:weak_sparse:--alpha 1.5"
    "weak_sparse_a2.0:weak_sparse:--alpha 2.0"
)

for model in "${MODELS[@]}"; do
  for sigma in "${SIGMAS[@]}"; do
    for beta_spec in "${BETAS[@]}"; do
      IFS=: read -r beta_tag preset alpha_flag <<< "$beta_spec"
      setting="${model}_sigma${sigma}_${beta_tag}"
      for norm in "${NORMS[@]}"; do
        arm="twostep_g${GAMMA}_${norm}"
        # $alpha_flag unquoted on purpose: empty expands to nothing.
        submit "eiv-${setting}-${arm}" "$OUT_ROOT/$setting/${arm}.json" \
               --model "$model" --n 100 --p 500 --sigma_a "$sigma" --sigma_e 3.0 \
               --beta "$preset" $alpha_flag --lam 1.0 \
               --algorithm twostep --gamma "$GAMMA" --norm "$norm" \
               --k 10 --true_k 3 --n_reps 100 --seed 0 --quiet --progress
      done
    done
  done
done

echo
echo "submitted: $n_submit   skipped (output already exists): $n_skip"
[[ $DRY_RUN -eq 1 ]] && echo "(dry run -- nothing was actually submitted)"
exit 0
