#!/bin/bash
# ==============================================================================
#   SLURM wrapper for the canonical paper run, scCODA_survival_FDR02.py.
#   128 GB, 16 cores, 12 hours, partition batch, account mpsnyder.
#
#   Environment notes that took a while to work out and will bite again:
#
#   PYTHONNOUSERSITE=1 keeps the shared home-directory site-packages out of the
#   path, which otherwise shadows the conda environment.
#
#   HOME is redirected to node-local /tmp. The per-user home quota blocks writes
#   during a run, and separately arviz fails to import silently when HOME is not
#   writable, which returns r-hat and effective sample size as NaN with no error. If
#   the diagnostics come back empty, this is why.
#
#   LD_LIBRARY_PATH must include the conda lib directory or JAX cannot find its
#   shared objects.
#
#   set -u comes only after conda is sourced. The conda profile script references
#   unbound variables, so enabling it first kills the job instantly with an empty
#   log.
# ==============================================================================
# Comments added 28 July 2026. Executable logic is byte-identical to the
# version that ran; a pristine copy sits in _verbatim/ and the annotator
# verifies the parse tree is unchanged.
# ==============================================================================

#SBATCH --job-name=scfdr02
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=mpsnyder

set -e  # keep errexit
# DO NOT set -u before sourcing conda (conda.sh may reference PS1)

mkdir -p logs

# Safest: ensure PS1 exists for conda.sh in batch shells
export PS1="${PS1-}"

source /scg/apps/software/miniconda/3/etc/profile.d/conda.sh
conda activate /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env

# Now it's safe to enable nounset if you want it
set -u
set -o pipefail

ENV_PY="$CONDA_PREFIX/bin/python"
SCRIPT="$(pwd)/scCODA_survival_FDR02.py"

echo "Host: $(hostname)"
echo "Conda prefix: $CONDA_PREFIX"
echo "Python: $ENV_PY"
"$ENV_PY" -V


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

"$ENV_PY" "$SCRIPT" 
