#!/bin/bash
# ==============================================================================
#   SLURM wrapper for milo_split_cohort_run.py Pineda2024 -- the external
#   Kellis cohort's half of the split-cohort Milo re-analysis, 192,111 cells.
#
#   Resource choices: 128G / 16 CPUs / 24-hour budget, roughly half the
#   memory and time of the companion ALS job (run_milo_split_ALS.sh), scaled
#   down for a third of the cell count. This job actually finished in 12
#   minutes 9 seconds, well inside the budget -- the generous allocation was
#   insurance against the same kind of non-linear scaling that made the ALS
#   job run longer than a naive extrapolation would predict, not a reflection
#   of how long this particular run needed.
#
#   Environment notes that will bite again on any future job using this
#   conda environment:
#
#   PYTHONNOUSERSITE=1 keeps the shared home-directory site-packages out of
#   the path, which otherwise shadows the conda environment's own h5py/numpy/
#   scanpy and silently pulls in incompatible versions.
#
#   LD_LIBRARY_PATH must include the conda lib directory or some of scanpy's
#   compiled dependencies can't find their shared objects.
#
#   set -u comes only after conda is sourced. The conda profile script itself
#   references unbound variables, so enabling it first kills the job
#   instantly with an empty log and no useful error.
# ==============================================================================
# Comments added 28 July 2026. Executable logic is byte-identical to the
# version that ran; a pristine copy sits in _verbatim/, and every non-comment
# line was confirmed unchanged with a plain diff before this copy shipped.
# ==============================================================================
#SBATCH --job-name=milosplit_Pineda
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=mpsnyder

set -e
mkdir -p logs
export PYTHONNOUSERSITE=1
export PS1="${PS1-}"

source /scg/apps/software/miniconda/3/etc/profile.d/conda.sh
conda activate /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env

set -u
set -o pipefail

ENV_PY="$CONDA_PREFIX/bin/python"
SCRIPT="$(pwd)/milo_split_cohort_run.py"

echo "Host: $(hostname)"
echo "Conda prefix: $CONDA_PREFIX"
"$ENV_PY" -V
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

"$ENV_PY" "$SCRIPT" Pineda2024
