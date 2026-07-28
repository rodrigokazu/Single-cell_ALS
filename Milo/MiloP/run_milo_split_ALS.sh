#!/bin/bash
# ==============================================================================
#   SLURM wrapper for milo_split_cohort_run.py ALS -- our own cohort's half of
#   the split-cohort Milo re-analysis, 586,219 cells, roughly three times
#   Pineda2024's cell count.
#
#   Resource choices: 256G / 16 CPUs / 40-hour budget mirrors the original
#   full-merged pipeline's own allocation (Milopy_survival.py's run_milo.sh),
#   kept generous on purpose. The companion Pineda2024 job at a third of the
#   cell count needed 128G and finished in 12 minutes; this one, at three
#   times the cells, took a good deal longer than a naive 3x extrapolation
#   from that would suggest (the make_nhoods step in particular scaled worse
#   than linearly), which is exactly the kind of surprise a generous time
#   budget is there to absorb without needing a resubmission.
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
#SBATCH --job-name=milosplit_ALS
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=40:00:00
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

"$ENV_PY" "$SCRIPT" ALS
