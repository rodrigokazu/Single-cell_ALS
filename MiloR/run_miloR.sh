#!/bin/bash

#SBATCH --job-name=milocomp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --account=mpsnyder

# --- Load R (R 4.3.3) on SCG ---
module load R/4.3.3

# Make sure personal library is used
export R_LIBS_USER=$HOME/Rlibs

export SCE_USE_ANNDATA_IO=1
export ZELLKONVERTER_USE_ANN_DATA_IO=TRUE
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH


# --- Run MiloR script ---
SCE_EXPORT_ALLOW_PLACEHOLDER=1 Rscript MiloR_PBMC_DA.R
