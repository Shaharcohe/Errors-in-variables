#!/bin/bash
# Submit the per-iteration tracking runs (track_iterations.py) to LSF.
#
#   ./submit_tracking.sh --dry-run    print the bsub lines, submit nothing
#   ./submit_tracking.sh              submit
#   ./submit_tracking.sh --smoke      one tiny job (2 seeds) to check the node env
#
# Same 24 settings, seeds and fixed parameters as submit_sweep.sh, but only the
# four plain arms cocolasso / reweighted x max / frobenius (no refit, no
# max_then_frobenius). Output: results/iteration_tracking/<setting>/<arm>.json.
#
# Calls python directly rather than through run_lsf.sh: that wrapper hardcodes
# REPO_DIR and PYTHON under /home/projects/nyosef/, which is not this checkout.
#
# reweighted+max is ~190 s/fit at p=500 and track_iterations.py writes only at
# the end, on a preemptable queue, so that arm is split into 4 x 25 seeds;
# plot_iterations.py merges the parts by seed.

set -euo pipefail

REPO_DIR="/home/projects/nadler/shachaco/Errors-in-variables"
# /usr/bin/python exists only on the login nodes. numpy, scipy and sparse_approx_gsm
# live in ~/.local/lib/python3.9, so any Python 3.9 on the node picks them up.
PY_MODULE="Python/3.9.6-GCCcore-11.2.0-bare"
PYTHON="python3"                # the module ships python3, not python
OUT_ROOT="results/iteration_tracking"
QUEUE="medium"
MEM=1000
WALL="12:00"

DRY_RUN=0
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --smoke)   SMOKE=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$REPO_DIR"
mkdir -p logs "$OUT_ROOT"

ENV_PREFIX="module load $PY_MODULE && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 &&"

n_submit=0
n_skip=0

submit() {          # $1 job name, $2 out json, rest: track_iterations.py args
    local jobname="$1" outfile="$2"
    shift 2
    if [[ -e "$outfile" ]]; then
        n_skip=$((n_skip + 1))
        return
    fi
    n_submit=$((n_submit + 1))
    local cmd=(bsub -q "$QUEUE" -n 1 -R "rusage[mem=$MEM]" -W "$WALL"
               -J "$jobname" -oo "logs/%J.out" -eo "logs/%J.err"
               "$ENV_PREFIX cd $REPO_DIR && $PYTHON track_iterations.py $* --out $outfile")
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
    else
        "${cmd[@]}"
    fi
}

if [[ $SMOKE -eq 1 ]]; then
    submit "eiv-track-smoke" "$OUT_ROOT/smoke/reweighted_frobenius.json" \
           --model ar --sigma_a 1.0 --beta datta_zhang \
           --algorithm reweighted --norm frobenius --seed 0 --n_reps 2
    echo "submitted: $n_submit   skipped: $n_skip"
    exit 0
fi

MODELS=(ar cs)
SIGMAS=(0.75 1.0 1.25)
CHUNK_SEEDS=(0 25 50 75)
ARMS=(cocolasso:max cocolasso:frobenius reweighted:max reweighted:frobenius)
# beta_tag:preset:alpha  -- datta_zhang ignores alpha.
BETAS=(
    "datta_zhang:datta_zhang:1.0"
    "weak_sparse_a1.0:weak_sparse:1.0"
    "weak_sparse_a1.5:weak_sparse:1.5"
    "weak_sparse_a2.0:weak_sparse:2.0"
)

for model in "${MODELS[@]}"; do
  for sigma in "${SIGMAS[@]}"; do
    for beta_spec in "${BETAS[@]}"; do
      IFS=: read -r beta_tag preset alpha <<< "$beta_spec"
      setting="${model}_sigma${sigma}_${beta_tag}"
      setting_dir="$OUT_ROOT/$setting"
      for arm_spec in "${ARMS[@]}"; do
        IFS=: read -r algorithm norm <<< "$arm_spec"
        arm="${algorithm}_${norm}"
        common=(--model "$model" --sigma_a "$sigma" --beta "$preset" --alpha "$alpha"
                --algorithm "$algorithm" --norm "$norm")
        if [[ "$algorithm" == reweighted && "$norm" == max ]]; then
            for s in "${CHUNK_SEEDS[@]}"; do
                submit "eiv-track-${setting}-${arm}-s${s}" \
                       "$setting_dir/parts/${arm}.seed${s}.json" \
                       "${common[@]}" --seed "$s" --n_reps 25
            done
        else
            submit "eiv-track-${setting}-${arm}" "$setting_dir/${arm}.json" \
                   "${common[@]}" --seed 0 --n_reps 100
        fi
      done
    done
  done
done

echo
echo "submitted: $n_submit   skipped (output already exists): $n_skip"
[[ $DRY_RUN -eq 1 ]] && echo "(dry run -- nothing was actually submitted)"
exit 0
