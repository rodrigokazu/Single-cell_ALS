#!/bin/bash
# ==============================================================================
#   SLURM wrapper for sccoda_multiome_age_wdr49.py, the authoritative contrast 7.
#   256 GB and 8 cores, though actual peak usage is around 1.5 GB since only obs is
#   read. Job 52251115 finished in 19 minutes 24 seconds.
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

#SBATCH --job-name=sccoda_mult_age_wdr49
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --account=mpsnyder

# Multiome-only (OurCohort) continuous-covariate contrasts 5/6/7 (survAdj,
# survival-only, age) -- completes the per-dataset table so "multiome" has all
# 7 contrasts, matching the integrated seven-contrast table. Pineda2024 is
# excluded (no Survival data at all; Age present in only 10/49 donors, all
# C9orf72 -- no cross-group comparison possible for that contrast).
# Reads obs only (dummy 1-column X), so memory is modest like run_perdataset.sh.

set -e
mkdir -p logs
export PS1="${PS1-}"

source /scg/apps/software/miniconda/3/etc/profile.d/conda.sh
conda activate /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env

set -u
set -o pipefail

export PYTHONNOUSERSITE=1
export HOME="/tmp/rk_home_${SLURM_JOB_ID:-0}"
export XDG_CACHE_HOME="$HOME/cache"
export MPLCONFIGDIR="$HOME/mpl"
export NUMBA_CACHE_DIR="$HOME/numba"
mkdir -p "$HOME/arviz_data" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ENV_PY="$CONDA_PREFIX/bin/python"
echo "Host: $(hostname)"; "$ENV_PY" -V
"$ENV_PY" "$(pwd)/sccoda_multiome_age_wdr49.py"
