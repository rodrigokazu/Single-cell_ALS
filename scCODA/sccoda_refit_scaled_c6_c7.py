#!/usr/bin/env python3
"""
Re-run scCODA contrasts (6) ALS_survival_only and (7) All_AgeAtDeath PROPERLY.

Fixes vs the original run:
  * Contrast 7 never completed (out-of-memory) -- run it here in isolation.
  * Continuous covariates (Survival, Age_At_Death) are STANDARDIZED (z-scored
    over donors) before fitting. Feeding them raw collapsed the posterior
    (SD=0, HDI width=0, inclusion prob=1 for every cell type) -> spurious
    "everything credible" calls. Standardizing restores a well-behaved fit.
  * Fixed rng seed for reproducibility.
  * Compact outputs only (effect tables + coda modality + posterior npz).
    We do NOT write the full 12-53 GB MuData (that + cumulative RAM caused the
    original OOM). Memory is freed between contrasts.

Outputs -> <OUTDIR>/<contrast>/:
  effect_df_<covariate>.csv, intercept_df.csv, credible_effects.csv,
  mean_composition_by_status.csv, per_sample_composition_wide.csv,
  coda_modality.h5ad, posterior_beta.npz, contrast_manifest.json
"""
# ==============================================================================
# WHAT THIS IS
#   The repair job for the three continuous-covariate contrasts that the paper run
#   got wrong. It refits contrasts 5, 6 and 7 with survival and age at death
#   standardised before they reach the model.
#
# WHY IT EXISTS
#   Feeding those covariates raw collapsed the posterior in the original run. The
#   fix is one line, the z-scoring below, and it restores well-behaved chains:
#   r-hat near 1.00 and effective sample sizes above 1,500.
#
# WHAT IT PRODUCES
#   .../sccoda/reviewer_refit_c6_c7_scaled/<contrast>/ , three subdirectories.
#
# WHERE IT LANDS IN THE SUPPLEMENTARY TABLE
#   Contrasts (v) and (vi) come from here and are authoritative. Its contrast 7 is
#   superseded, not because the fit was bad but because the cohort scope was wrong,
#   which sccoda_multiome_age_wdr49.py fixes.
#
# HOW IT PICKS DONORS, AND THE TRAP IN THAT
#   This script does not filter on cohort. It drops rows whose covariate is missing,
#   which you can see at the nan_mask line, and it lands on our 35 donors only
#   because every external donor is missing survival. The restriction is a side
#   effect rather than a decision. If the external metadata ever gains a survival
#   field, this script will silently start pooling both cohorts with no warning and
#   no code change. Add an explicit study filter before rerunning it.
#
#   Survival is recorded for no control donor anywhere, in either cohort. That is
#   not an oversight in the data: survival is months after diagnosis, and a control
#   has no diagnosis to count from. So a survival-adjusted comparison against
#   controls cannot be fitted at all, which is why no such contrast exists.
# ==============================================================================
# Comments added 28 July 2026. Executable logic is byte-identical to the
# version that ran; a pristine copy sits in _verbatim/ and the annotator
# verifies the parse tree is unchanged.
# ==============================================================================


import warnings
warnings.filterwarnings("ignore")

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)

import gc
import sys
import json
import time
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import pertpy as pt


# ---------------------------------------------------------------------------
IN_H5AD = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Final_merged_h5ad/all_cells_scvi_integrated.h5ad"
OUTDIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/sccoda/reviewer_refit_c6_c7_scaled"

CELL_TYPE_COL = "celltype_lr_full"
SAMPLE_COL = "orig.ident"
STATUS_SOURCE_COL = "type_als"
SURVIVAL_COL = "Survival"
AGE_COL = "Age_At_Death"

FDR_LEVEL = 0.2
RNG_KEY = 42
NUM_SAMPLES = 10000
NUM_WARMUP = 1000


@dataclass
class Contrast:
    number: int
    name: str
    label: str
    keep_statuses: list
    control_level: str
    raw_covariate: str          # continuous covariate to standardize
    include_status_term: bool   # whether status_ref is in the formula


CONTRASTS = [
    # Contrast 5 (survAdj) is ALSO degenerate as-run (SD=0, rhat 1.7-2.6 on BOTH
    # its status and Survival covariates), so it is refit here too. Comment it out
    # to restrict the job to contrasts 6 and 7 only.
    Contrast(5, "C9orf72_vs_Sporadic_survAdj",
             "C9orf72 vs Sporadic, Survival (standardized) adjusted",
             ["C9orf72", "Sporadic"], "Sporadic", SURVIVAL_COL, include_status_term=True),
    Contrast(6, "ALS_survival_only",
             "ALS donors: Survival (standardized) as sole covariate",
             ["C9orf72", "Sporadic"], "Sporadic", SURVIVAL_COL, include_status_term=False),
    Contrast(7, "All_AgeAtDeath",
             "All donors: Age at death (standardized) as covariate",
             ["C9orf72", "Sporadic", "Control"], "Control", AGE_COL, include_status_term=True),
]


def _flush(*a, **k):
    print(*a, **k); sys.stdout.flush()


def _as_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def _force_control_reference(coda, status_col, control_level, keep_statuses):
    obs = coda.obs
    present = sorted(obs[status_col].astype(str).unique().tolist())
    ordered = [control_level] + [x for x in keep_statuses if x != control_level and x in present]
    extras = [x for x in present if x not in ordered]
    obs["status_ref"] = pd.Categorical(obs[status_col].astype(str),
                                       categories=ordered + extras, ordered=True)
    coda.obs = obs
    return "status_ref"


def _composition_by_status(coda, status_col, control_level, keep_statuses):
    X = _as_dense(coda.X).astype(float)
    rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    comp = X / rs
    cts = list(coda.var_names.astype(str))
    wide = pd.DataFrame(comp, index=coda.obs_names.astype(str), columns=cts)
    wide["status"] = coda.obs[status_col].astype(str).values
    order = [control_level] + [s for s in keep_statuses if s != control_level]
    mean_df = wide.groupby("status", observed=True).mean(numeric_only=True)
    mean_df = mean_df.loc[[s for s in order if s in mean_df.index]]
    return mean_df, wide


def run_contrast(adata, con, outroot):
    _flush(f"\n{'='*70}\n[CONTRAST {con.number}] {con.name}: {con.label}\n{'='*70}")
    cdir = Path(outroot) / con.name
    cdir.mkdir(parents=True, exist_ok=True)
    man = {"number": con.number, "name": con.name, "label": con.label,
           "raw_covariate": con.raw_covariate, "rng_key": RNG_KEY,
           "fdr_level": FDR_LEVEL, "notes": []}

    # subset donors
    status = adata.obs[STATUS_SOURCE_COL].astype(str)
    sub = adata[status.isin(con.keep_statuses).values].copy()

    # drop cells with missing covariate
    cov = con.raw_covariate
    sub.obs[cov] = pd.to_numeric(sub.obs[cov], errors="coerce")
    # Donor selection happens here, and it is a missing-data drop rather than a
    # cohort filter. It lands on our 35 donors only because no external donor has
    # survival recorded. Emergent, not declared: see the header warning.
    nan_mask = sub.obs[cov].isna()
    if nan_mask.any():
        _flush(f"  dropping {int(nan_mask.sum())} cells with NaN {cov}")
        sub = sub[~nan_mask.values].copy()

    # STANDARDIZE covariate over donors (unweighted by cell count)
    per_donor = sub.obs.groupby(SAMPLE_COL)[cov].first()
    # This is the line the script exists for. Standardising on donor-level moments, not
    # cell-level, so one donor contributes once regardless of nuclei recovered.
    # Effects are therefore per standard deviation of the covariate: 24.78 months
    # for survival, 9.83 years for age.
    mu, sigma = float(per_donor.mean()), float(per_donor.std(ddof=0))
    if sigma == 0 or not np.isfinite(sigma):
        raise ValueError(f"{cov} has zero/invalid variance across donors")
    covz = f"{cov}_z"
    sub.obs[covz] = (sub.obs[cov].astype(float) - mu) / sigma
    man.update({"covariate_z": covz, "donor_mean": mu, "donor_sd": sigma,
                "n_donors": int(per_donor.shape[0]), "n_cells": int(sub.n_obs)})
    _flush(f"  standardized {cov}: donor mean={mu:.3f} sd={sigma:.3f} "
           f"({per_donor.shape[0]} donors, {sub.n_obs} cells)")

    status_key = f"_cstatus_{con.name}"
    sub.obs[status_key] = sub.obs[STATUS_SOURCE_COL].astype(str)

    # build scCODA data
    model = pt.tl.Sccoda()
    covariate_obs = [status_key, covz]
    data = model.load(sub, type="cell_level", generate_sample_level=True,
                      cell_type_identifier=CELL_TYPE_COL, sample_identifier=SAMPLE_COL,
                      covariate_obs=covariate_obs)
    mod_key = f"coda_{con.name}"
    coda = data["coda"]
    mask = coda.obs[status_key].astype(str).isin(con.keep_statuses).values
    data.mod[mod_key] = coda[mask].copy()
    n_s, n_ct = data.mod[mod_key].n_obs, data.mod[mod_key].n_vars
    _flush(f"  coda: {n_s} donors x {n_ct} cell types")
    man["n_samples"], man["n_celltypes"] = int(n_s), int(n_ct)

    _force_control_reference(data.mod[mod_key], status_key, con.control_level, con.keep_statuses)
    formula = (f"status_ref + {covz}") if con.include_status_term else f"{covz}"
    man["formula"] = formula
    _flush(f"  formula: {formula}")

    data = model.prepare(data, modality_key=mod_key, formula=formula)
    _flush(f"  running NUTS (seed={RNG_KEY}, {NUM_SAMPLES} samples) ...")
    model.run_nuts(data, modality_key=mod_key, num_samples=NUM_SAMPLES,
                   num_warmup=NUM_WARMUP, rng_key=RNG_KEY)

    # credible effects
    try:
        cred = model.credible_effects(data, modality_key=mod_key, est_fdr=FDR_LEVEL)
        cred.to_csv(cdir / "credible_effects.csv")
    except Exception as e:
        man["notes"].append(f"credible_effects failed: {e!r}")

    # effect / intercept tables + degeneracy check
    coda_fit = data[mod_key]
    ref_ct = coda_fit.uns.get("scCODA_params", {}).get("reference_cell_type")
    man["reference_cell_type"] = str(ref_ct)
    degenerate_any = False
    for k in list(coda_fit.varm.keys()):
        df = coda_fit.varm[k]
        if isinstance(df, pd.DataFrame):
            df.to_csv(cdir / f"{k}.csv")
            if k.startswith("effect_df_"):
                width = (pd.to_numeric(df["HDI 97%"], errors="coerce")
                         - pd.to_numeric(df["HDI 3%"], errors="coerce")).abs()
                sd = pd.to_numeric(df["SD"], errors="coerce").abs()
                deg = bool((sd.fillna(0) < 1e-9).all() and (width.fillna(0) < 1e-9).all())
                degenerate_any = degenerate_any or deg
                _flush(f"  {k}: n_credible={(df['Final Parameter']!=0).sum()}/{len(df)} "
                       f"degenerate={deg}")
    man["posterior_degenerate"] = degenerate_any

    # compositions
    mean_df, wide = _composition_by_status(coda_fit, status_key, con.control_level, con.keep_statuses)
    mean_df.to_csv(cdir / "mean_composition_by_status.csv")
    wide.to_csv(cdir / "per_sample_composition_wide.csv")

    # compact posterior for diagnostics
    try:
        samp = coda_fit.uns["scCODA_params"]["mcmc"]["samples"]
        np.savez_compressed(cdir / "posterior_beta.npz",
                            beta=np.asarray(samp["beta"]),
                            covariate_names=np.asarray(
                                coda_fit.uns["scCODA_params"]["covariate_names"]))
    except Exception as e:
        man["notes"].append(f"posterior save failed: {e!r}")

    # small coda modality only (NOT the full MuData)
    try:
        coda_fit.write(cdir / "coda_modality.h5ad")
    except Exception as e:
        man["notes"].append(f"coda h5ad write failed: {e!r}")

    with open(cdir / "contrast_manifest.json", "w") as f:
        json.dump(man, f, indent=2, default=str)
    _flush(f"  [DONE] contrast {con.number} -> {cdir}")

    # free memory before next contrast
    del sub, data, model, coda, coda_fit
    gc.collect()


def main():
    t0 = time.time()
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    _flush(f"[INFO] reading {IN_H5AD}")
    adata = sc.read(IN_H5AD)
    adata.obs[STATUS_SOURCE_COL] = adata.obs[STATUS_SOURCE_COL].astype(str)

    # strip heavy fields to avoid memory blowup on subset
    for attr in ("obsp", "obsm", "varm", "varp"):
        obj = getattr(adata, attr, None)
        if obj is not None:
            for k in list(obj.keys()):
                del obj[k]
    if len(adata.uns) > 0:
        adata.uns.clear()
    gc.collect()
    _flush(f"[INFO] {adata.n_obs} cells, {adata.obs[CELL_TYPE_COL].nunique()} cell types")

    for con in CONTRASTS:
        try:
            run_contrast(adata, con, OUTDIR)
        except Exception as e:
            import traceback
            _flush(f"[ERROR] contrast {con.number} {con.name}: {e!r}")
            traceback.print_exc()

    _flush(f"\n[INFO] all done in {time.time()-t0:.1f}s -> {OUTDIR}")


if __name__ == "__main__":
    main()
