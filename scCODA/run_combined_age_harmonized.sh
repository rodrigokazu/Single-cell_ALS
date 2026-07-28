#!/bin/bash
# ==============================================================================
#   SLURM wrapper for sccoda_combined_age_harmonized.py, the contrast 7 sensitivity
#   check. Same shape as its sibling. Job 52259919 finished in 29 minutes.
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

#SBATCH --job-name=sccoda_comb_age_harm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --account=mpsnyder

# Combined-dataset (ALS/multiome + Pineda2024/Kellis) contrast 7 only, with
# "Astrocytes WDR49" harmonized into "Astrocytes" so Pineda2024's 10 donors
# with valid Age_At_Death (all C9orf72) can be included without forcing the
# WDR49-zero artefact. Companion run to sccoda_multiome_age_wdr49.py.
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
"$ENV_PY" "$(pwd)/sccoda_combined_age_harmonized.py"
