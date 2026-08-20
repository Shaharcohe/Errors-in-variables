#!/bin/bash
# LSF batch job: run main.py's simulation sweep on a CPU queue.
#
# Any arguments are forwarded to main.py verbatim, e.g.:
#
#   bsub -q short -n 1 -R "rusage[mem=4000]" -W 02:00 \
#       -o logs/%J.out -e logs/%J.err \
#       ./run_lsf.sh --p 60 --n_reps 20 --algorithm cocolasso,reweighted
#
# The #BSUB defaults below are only picked up by LSF when this file is
# submitted with NO extra arguments after it (`bsub run_lsf.sh` or
# `bsub < run_lsf.sh`); once arguments are appended for main.py, LSF treats
# the whole thing as a plain command line and ignores the embedded
# directives, silently falling back to the site default queue. So always
# pass -q (and any other options you care about) on the bsub command line
# as above, rather than relying on the block below.
#
# Submitting with no main.py arguments at all -- e.g. `bsub < run_lsf.sh`,
# which does not forward argv -- runs the quick end-to-end check from the
# README instead of falling into main.py's interactive prompt (which would
# otherwise hang forever with no terminal attached).
#
#BSUB -q short
#BSUB -J eiv-test
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 02:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail

REPO_DIR="/home/projects/nyosef/shachaco/Errors-in-variables"
GSM_SRC="/home/projects/nyosef/shachaco/gsm/sparse-approximation-gsm/sparse-approx-gsm"
PYTHON="/home/projects/nyosef/shachaco/miniconda3/envs/nolan_env/bin/python"

# eiv_algorithm.py imports sparse_approx_gsm. It is pip-installed in
# nolan_env, but that install packaged only its compiled CUDA extension, not
# the pure-Python sources that define gsm() -- so it must come from the
# sibling source checkout instead.
export PYTHONPATH="$GSM_SRC${PYTHONPATH:+:$PYTHONPATH}"

# Match BLAS/OpenMP threads to the LSF slot count (-n above) to avoid
# oversubscribing the host; raise both if -n is increased.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$REPO_DIR"
mkdir -p logs results

if [ "$#" -eq 0 ]; then
    set -- --p 60 --n_reps 5 --algorithm cocolasso,reweighted
fi

echo "host: $(hostname)"
echo "python: $PYTHON"
echo "args: $*"

exec "$PYTHON" main.py "$@"
