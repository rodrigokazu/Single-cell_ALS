#!/bin/bash
# ==============================================================================
#   SLURM wrapper for sccoda_refit_scaled_c6_c7.py, the z-scored refit of contrasts
#   5, 6 and 7. 256 GB, raised from the 128 GB that ran out of memory on the
#   original attempt, and it writes compact outputs only rather than the 12 to 53 GB
#   full MuData per contrast that caused it.
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

#SBATCH --job-name=sccoda_refit_c6c7
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --account=mpsnyder

# scCODA re-run of contrasts (6) ALS_survival_only and (7) All_AgeAtDeath
# with STANDARDIZED continuous covariates + fixed seed + compact outputs.
# 256G (vs 128G) + no giant MuData write fixes the original OOM.
#
# STAGED: submit with `sbatch run_refit_c6_c7.sh` when ready.

set -e
mkdir -p logs
export PS1="${PS1-}"

source /scg/apps/software/miniconda/3/etc/profile.d/conda.sh
conda activate /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env

set -u
set -o pipefail

export PYTHONNOUSERSITE=1
# home has a per-user quota block -> redirect HOME + all caches to node-local /tmp
export HOME="/tmp/rk_home_${SLURM_JOB_ID:-0}"
export XDG_CACHE_HOME="$HOME/cache"
export MPLCONFIGDIR="$HOME/mpl"
export NUMBA_CACHE_DIR="$HOME/numba"
mkdir -p "$HOME/arviz_data" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ENV_PY="$CONDA_PREFIX/bin/python"
SCRIPT="$(pwd)/sccoda_refit_scaled_c6_c7.py"

echo "Host: $(hostname)"
echo "Python: $ENV_PY"
"$ENV_PY" -V

"$ENV_PY" "$SCRIPT"
