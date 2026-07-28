#!/usr/bin/env python3
"""
Multiome-only (OurCohort, study=="ALS") continuous-covariate scCODA contrasts:
  (5) C9orf72 vs Sporadic, Survival (standardized) adjusted
  (6) ALS donors (C9orf72+Sporadic): Survival (standardized) as sole covariate
  (7) All donors: Age at death (standardized) as covariate

Completes the per-dataset table so "multiome" has all 7 contrasts, matching the
integrated (combined) dataset's seven-contrast table. Pineda2024 is EXCLUDED from
this script by design: it has zero donors with Survival data, and only 10/49
donors with Age at death -- all of them C9orf72, so no Control/Sporadic arm
exists for the age contrast. That is a hard data-availability gap, not a power
issue, and is reported as N/A rather than run.

Mirrors sccoda_perdataset.py's lightweight obs-only load (dummy 1-column X, no
16GB counts layer) and sccoda_refit_scaled_c6_c7.py's donor-level z-scoring of
continuous covariates (feeding them raw collapsed the posterior in the original
run: SD=0, HDI width=0, inclusion prob=1 for every cell type).

Cell types HARMONIZED to the 21-type shared schema ("Astrocytes WDR49" merged
into "Astrocytes") for consistency with the other per-dataset contrasts (1-4);
the WDR49-resolved numbers live in the integrated seven-contrast table.

Outputs -> OUTDIR/<contrast_name>/: effect_df_<cov>.csv, credible_effects.csv,
  mean_composition_by_status.csv, per_sample_composition_wide.csv,
  coda_modality.h5ad, posterior_beta.npz, contrast_manifest.json
  + OUTDIR/multiome_continuous_long.csv, multiome_continuous_manifest.json
"""
# ==============================================================================
# WHAT THIS IS
#   Contrasts 5, 6 and 7 fitted on our dataset alone with cell types harmonised to
#   the 21 shared labels. Built so the per-dataset table would have all seven
#   contrasts for our cohort, matching the schema of the categorical split.
#
# WHAT IT PRODUCES
#   .../sccoda/reviewer_perdataset_continuous/OurCohort_multiome/
#
# WHERE IT LANDS
#   It supported the three-way comparison spreadsheet sent in July. It is NOT the
#   source of any row in Supplementary_Table_scCODA_seven_contrasts_FINAL.xlsx.
#   Contrasts (v) and (vi) there come from sccoda_refit_scaled_c6_c7.py, and (vii)
#   from sccoda_multiome_age_wdr49.py.
#
# WHY IT CANNOT ANSWER THE WDR49 QUESTION
#   HARMONIZE_WDR49 is true here, so WDR49+ astrocytes are folded into astrocytes
#   and 21 cell types are tested. Consistency with the categorical per-dataset run
#   was the point, but it means this script is silent on the one population the
#   paper cares most about. That is what prompted sccoda_multiome_age_wdr49.py.
#
# THE JOB SAGA, WHICH WILL RECUR
#   Four submissions were needed. The first three died mid-sampling with a CPython
#   fatal error, ruled out as neither memory nor a bad node: peak usage never passed
#   1.5 GB, and the same signature appeared on three different machines. The cause
#   was tqdm's progress bar racing JAX's threaded backend. The fourth attempt
#   disabled both and finished in 24 minutes 30 seconds.
# ==============================================================================
# Comments added 28 July 2026. Executable logic is byte-identical to the
# version that ran; a pristine copy sits in _verbatim/ and the annotator
# verifies the parse tree is unchanged.
# ==============================================================================


import warnings; warnings.filterwarnings("ignore")
import os
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
# 3 prior attempts all died mid-NUTS-sampling with a CPython fatal error
# ("Fatal Python error: deallocating None" / object address+refcount+type dump,
# no traceback) at unpredictable sample indices (11-72%) on continuous-covariate
# contrasts only -- a known class of bug from tqdm's host_callback progress-bar
# updater racing with JAX's multi-threaded CPU (Eigen) backend. Disable both.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc, sys, json, time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import anndata as ad
import pertpy as pt
import numpyro.infer as _npyinfer
import pertpy.tools._coda._base_coda as _base_coda

class _MCMCNoProgress(_npyinfer.MCMC):
    def __init__(self, *a, **kw):
        kw.setdefault("progress_bar", False)
        super().__init__(*a, **kw)

_base_coda.MCMC = _MCMCNoProgress  # kill the tqdm host_callback updater

try:
    import arviz as az; _HAVE_ARVIZ = True
except Exception:
    _HAVE_ARVIZ = False

IN_H5AD = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Final_merged_h5ad/all_cells_scvi_integrated.h5ad"
OUTDIR  = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/sccoda/reviewer_perdataset_continuous"
DS_LABEL = "OurCohort_multiome"
STUDY_VALUE = "ALS"

CELL_TYPE_COL = "celltype_lr_full"
SAMPLE_COL    = "orig.ident"
STATUS_COL    = "type_als"
STUDY_COL     = "study"
SURVIVAL_COL  = "Survival"
AGE_COL       = "Age_At_Death"
FDR_LEVEL     = 0.2
RNG_KEY       = 42
NUM_SAMPLES   = 10000
NUM_WARMUP    = 1000
# True, so WDR49+ astrocytes are folded into astrocytes and 21 cell types are
# tested. Consistent with the categorical per-dataset run, and the reason this
# script cannot address WDR49 at all.
HARMONIZE_WDR49 = True

SHORT = {"Astrocytes":"ASC","Claustrum Neurons":"CLA","Corticothalamic L6":"CT L6",
    "Endothelial Cells":"Endo","Excitatory L5":"ET L5","Interneuron L2/3":"IT L2/3",
    "Interneuron L4":"IT L4","Interneuron L5":"IT L5","Interneuron L6":"IT L6",
    "Layer 6b Neurons":"L6b","LAMP5 Interneurons":"LAMP5","Microglia":"MGC",
    "NDNF Interneurons":"NDNF","Near-Projecting Neurons":"NP","Oligodendrocytes":"ODC",
    "Oligodendrocyte Precursor Cells":"OPC","Parvalbumin Interneurons":"PVALB",
    "Parvalbumin Chandelier Cells":"PVALB ChC","Somatostatin-Expressing Neurons":"SST",
    "VIP Interneurons":"VIP","Vascular Leptomeningeal Cells":"VLMC","Astrocytes WDR49":"ASC WDR49"}


@dataclass
class Contrast:
    number: int
    name: str
    label: str
    keep_statuses: list
    control_level: str
    raw_covariate: str
    include_status_term: bool


CONTRASTS = [
    Contrast(5, "C9orf72_vs_Sporadic_survAdj",
             "C9orf72 vs Sporadic, Survival (standardized) adjusted",
             ["C9orf72", "Sporadic"], "Sporadic", SURVIVAL_COL, True),
    Contrast(6, "ALS_survival_only",
             "ALS donors: Survival (standardized) as sole covariate",
             ["C9orf72", "Sporadic"], "Sporadic", SURVIVAL_COL, False),
    Contrast(7, "All_AgeAtDeath",
             "All donors: Age at death (standardized) as covariate",
             ["C9orf72", "Sporadic", "Control"], "Control", AGE_COL, True),
]


def _flush(*a, **k): print(*a, **k); sys.stdout.flush()


def _read_obs(path):
    f = h5py.File(path, "r"); og = f["obs"]
    want = [CELL_TYPE_COL, SAMPLE_COL, STATUS_COL, STUDY_COL, SURVIVAL_COL, AGE_COL]
    def col(name):
        e = og[name]
        if isinstance(e, h5py.Group):
            cats = e["categories"][...]; codes = e["codes"][...]
            cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
            return np.where(codes >= 0, cats[codes.clip(min=0)], "NaN").astype(str)
        arr = e[...]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
    idx = col("_index") if "_index" in og else np.arange(len(og[want[0]]["codes"]) if isinstance(og[want[0]], h5py.Group) else len(og[want[0]]))
    obs = pd.DataFrame({c: col(c) for c in want}, index=pd.Index(idx, name="cell"))
    f.close()
    return obs


def _split_rhat_ess(x):
    if not _HAVE_ARVIZ: return (np.nan, np.nan)
    try:
        x = np.asarray(x, float).ravel()
        if x.size < 8 or np.nanstd(x) == 0: return (np.nan, np.nan)
        d = x.size // 2
        chains = np.stack([x[:d], x[d:2*d]])
        return (float(az.rhat(chains)), float(az.ess(chains)))
    except Exception:
        return (np.nan, np.nan)


def _composition(coda, status_key, control, keep):
    X = coda.X.toarray() if sp.issparse(coda.X) else np.asarray(coda.X)
    X = X.astype(float); rs = X.sum(1, keepdims=True); rs[rs == 0] = 1.0
    wide = pd.DataFrame(X / rs, index=coda.obs_names.astype(str), columns=list(coda.var_names.astype(str)))
    wide["status"] = coda.obs[status_key].astype(str).values
    order = [control] + [s for s in keep if s != control]
    mean = wide.groupby("status", observed=True).mean(numeric_only=True)
    mean = mean.loc[[s for s in order if s in mean.index]]
    return mean, wide


def run_one(obs_all, con, outroot):
    cdir = outroot / DS_LABEL / con.name; cdir.mkdir(parents=True, exist_ok=True)
    man = {"dataset": DS_LABEL, "number": con.number, "name": con.name, "label": con.label,
           "raw_covariate": con.raw_covariate, "fdr_level": FDR_LEVEL, "rng_key": RNG_KEY, "notes": []}

    sub = obs_all[obs_all[STATUS_COL].isin(con.keep_statuses)].copy()
    cov = con.raw_covariate
    sub[cov] = pd.to_numeric(sub[cov], errors="coerce")
    nan_mask = sub[cov].isna()
    if nan_mask.any():
        _flush(f"  dropping {int(nan_mask.sum())} cells with NaN {cov}")
        sub = sub[~nan_mask].copy()

    per_donor = sub.groupby(SAMPLE_COL)[cov].first()
    donors_per_status = sub.groupby(STATUS_COL)[SAMPLE_COL].nunique().to_dict()
    man["donors_per_group"] = donors_per_status
    if len(per_donor) < 6 or any(v < 3 for v in donors_per_status.values()):
        man["notes"].append(f"SKIP: too few donors with {cov} data {donors_per_status}")
        _flush(f"    [SKIP] {DS_LABEL}/{con.name}: donors {donors_per_status}")
        json.dump(man, open(cdir / "contrast_manifest.json", "w"), indent=2, default=str)
        return None

    mu, sigma = float(per_donor.mean()), float(per_donor.std(ddof=0))
    covz = f"{cov}_z"
    sub[covz] = (sub[cov].astype(float) - mu) / sigma
    man.update({"covariate_z": covz, "donor_mean": mu, "donor_sd": sigma, "n_donors": int(len(per_donor))})
    _flush(f"    {DS_LABEL}/{con.name}: standardized {cov} mean={mu:.3f} sd={sigma:.3f} "
           f"({len(per_donor)} donors); donors_per_status {donors_per_status}")

    status_key = f"_cstatus_{con.name}"
    sub[status_key] = sub[STATUS_COL].astype(str)
    A = ad.AnnData(X=sp.csr_matrix((sub.shape[0], 1), dtype="float32"), obs=sub)

    model = pt.tl.Sccoda()
    covariate_obs = [status_key, covz]
    data = model.load(A, type="cell_level", generate_sample_level=True,
                      cell_type_identifier=CELL_TYPE_COL, sample_identifier=SAMPLE_COL,
                      covariate_obs=covariate_obs)
    mod = f"coda_{con.name}"
    coda = data["coda"]
    m = coda.obs[status_key].astype(str).isin(con.keep_statuses).values
    data.mod[mod] = coda[m].copy()
    n_s, n_ct = data.mod[mod].n_obs, data.mod[mod].n_vars
    man["n_samples"], man["n_celltypes"] = int(n_s), int(n_ct)

    o = data.mod[mod].obs
    present = sorted(o[status_key].astype(str).unique())
    cats = [con.control_level] + [x for x in con.keep_statuses if x != con.control_level and x in present]
    cats += [x for x in present if x not in cats]
    o["status_ref"] = pd.Categorical(o[status_key].astype(str), categories=cats, ordered=True)
    data.mod[mod].obs = o

    formula = (f"status_ref + {covz}") if con.include_status_term else f"{covz}"
    man["formula"] = formula
    _flush(f"    formula: {formula}")
    data = model.prepare(data, modality_key=mod, formula=formula)
    model.run_nuts(data, modality_key=mod, num_samples=NUM_SAMPLES,
                   num_warmup=NUM_WARMUP, rng_key=RNG_KEY)

    coda_fit = data[mod]
    params = coda_fit.uns.get("scCODA_params", {})
    ref_ct = params.get("reference_cell_type"); man["reference_cell_type"] = str(ref_ct)
    cov_names = [str(x) for x in np.asarray(params.get("covariate_names", [])).ravel().tolist()]
    beta = None
    try: beta = np.asarray(params["mcmc"]["samples"]["beta"])
    except Exception: pass

    try:
        cred = model.credible_effects(data, modality_key=mod, est_fdr=FDR_LEVEL)
        cred.to_csv(cdir / "credible_effects.csv")
    except Exception as e:
        man["notes"].append(f"credible_effects failed: {e!r}")
    intr, eff = model.summary_prepare(coda_fit, est_fdr=FDR_LEVEL)

    rows = []
    per_cov = {}
    if isinstance(eff.index, pd.MultiIndex):
        for cv, s in eff.groupby(level=0):
            s = s.copy(); s.index = s.index.get_level_values(-1).astype(str); per_cov[str(cv)] = s
    else:
        per_cov[cov_names[0] if cov_names else "effect"] = eff.copy()

    # keep only the continuous covariate row (drop status_ref nuisance term for the
    # multiome_continuous_long.csv headline table, but write both to effect_df CSVs)
    for cv, df in per_cov.items():
        df.to_csv(cdir / f"effect_df_{cv}.csv")
        sd = pd.to_numeric(df.get("SD"), errors="coerce")
        lo = pd.to_numeric(df.get("HDI 3%"), errors="coerce")
        hi = pd.to_numeric(df.get("HDI 97%"), errors="coerce")
        incl = pd.to_numeric(df.get("Inclusion probability"), errors="coerce")
        width = (hi - lo).abs()
        rmap, emap = {}, {}
        if beta is not None and cv in cov_names:
            ci = cov_names.index(cv)
            if beta.ndim == 3 and ci < beta.shape[1] and beta.shape[2] == df.shape[0]:
                for j, ct in enumerate(df.index):
                    rmap[ct], emap[ct] = _split_rhat_ess(beta[:, ci, j])
        rser = pd.to_numeric(pd.Series(rmap), errors="coerce").dropna()
        frac_col = float(((sd.fillna(1).abs() < 1e-9) & (width.fillna(1) < 1e-9)).mean())
        frac_bad = float((rser > 1.1).mean()) if len(rser) else 0.0
        max_rhat = float(rser.max()) if len(rser) else np.nan
        degen = bool(frac_col >= 0.8 or frac_bad >= 0.2 or (np.isfinite(max_rhat) and max_rhat > 1.2))
        is_primary = (cv == covz)
        for ct in df.index:
            fp = float(pd.to_numeric(df.loc[ct, "Final Parameter"], errors="coerce"))
            rows.append({
                "dataset": DS_LABEL, "contrast_num": con.number, "contrast": con.name,
                "contrast_label": con.label, "covariate": cv, "is_primary_covariate": is_primary,
                "cell_type": ct, "cell_type_short": SHORT.get(ct, ct),
                "reference_cell_type": str(ref_ct),
                "log2_fold_change": float(pd.to_numeric(df.loc[ct, "log2-fold change"], errors="coerce")),
                "final_parameter": fp, "HDI_3pct": float(lo.get(ct, np.nan)),
                "HDI_97pct": float(hi.get(ct, np.nan)), "SD": float(sd.get(ct, np.nan)),
                "inclusion_probability": float(incl.get(ct, np.nan)),
                "credible_FDR0.2": bool(abs(fp) > 0),
                "rhat": float(rmap.get(ct, np.nan)), "ess_bulk": float(emap.get(ct, np.nan)),
                "posterior_degenerate": degen, "reliable": (not degen),
            })
    man["n_credible_primary"] = int(sum(r["credible_FDR0.2"] for r in rows if r["is_primary_covariate"]))

    mean, wide = _composition(coda_fit, status_key, con.control_level, con.keep_statuses)
    mean.to_csv(cdir / "mean_composition_by_status.csv")
    wide.to_csv(cdir / "per_sample_composition_wide.csv")
    try:
        np.savez_compressed(cdir / "posterior_beta.npz", beta=beta if beta is not None else np.array([]),
                            covariate_names=np.asarray(cov_names))
        coda_fit.write(cdir / "coda_modality.h5ad")
    except Exception as e:
        man["notes"].append(f"artifact save failed: {e!r}")
    json.dump(man, open(cdir / "contrast_manifest.json", "w"), indent=2, default=str)
    _flush(f"      -> credible(primary cov)@FDR0.2 = {man['n_credible_primary']} (ref={ref_ct})")

    del A, data, model, coda, coda_fit; gc.collect()
    return rows


def main():
    t0 = time.time(); out = Path(OUTDIR); out.mkdir(parents=True, exist_ok=True)
    _flush(f"[INFO] reading obs from {IN_H5AD}")
    obs = _read_obs(IN_H5AD)
    obs = obs[obs[STUDY_COL] == STUDY_VALUE].copy()
    _flush(f"[INFO] {DS_LABEL} ({STUDY_VALUE}): {len(obs)} cells, {obs[SAMPLE_COL].nunique()} donors")

    if HARMONIZE_WDR49:
        n = int((obs[CELL_TYPE_COL] == "Astrocytes WDR49").sum())
        obs[CELL_TYPE_COL] = obs[CELL_TYPE_COL].replace({"Astrocytes WDR49": "Astrocytes"})
        _flush(f"[INFO] harmonized {n} 'Astrocytes WDR49' cells into 'Astrocytes' "
               f"(-> {obs[CELL_TYPE_COL].nunique()} shared cell types)")

    all_rows = []
    manifest = {"in_h5ad": IN_H5AD, "outdir": OUTDIR, "dataset": DS_LABEL, "study_value": STUDY_VALUE,
                "fdr_level": FDR_LEVEL, "harmonize_wdr49": HARMONIZE_WDR49, "rng_key": RNG_KEY,
                "n_cells": int(len(obs)), "n_donors": int(obs[SAMPLE_COL].nunique()), "runs": []}

    for con in CONTRASTS:
        man_path = out / DS_LABEL / con.name / "contrast_manifest.json"
        long_prev = out / "multiome_continuous_long.csv"
        if man_path.exists():
            prev = json.loads(man_path.read_text())
            fatal_notes = [n for n in prev.get("notes", []) if "SKIP" in n or "credible_effects failed" in n]
            if "n_credible_primary" in prev and not fatal_notes:
                _flush(f"[RESUME] {con.name} already completed (n_credible_primary="
                       f"{prev['n_credible_primary']}) -- reloading from disk, skipping re-run")
                eff_files = sorted((out / DS_LABEL / con.name).glob("effect_df_*.csv"))
                rows_prev = []
                for fp in eff_files:
                    cv = fp.stem.replace("effect_df_", "")
                    df = pd.read_csv(fp, index_col=0)
                    is_primary = cv == prev.get("covariate_z")
                    for ct, r in df.iterrows():
                        rows_prev.append({
                            "dataset": DS_LABEL, "contrast_num": con.number, "contrast": con.name,
                            "contrast_label": con.label, "covariate": cv, "is_primary_covariate": is_primary,
                            "cell_type": ct, "cell_type_short": SHORT.get(ct, ct),
                            "reference_cell_type": prev.get("reference_cell_type"),
                            "log2_fold_change": float(r["log2-fold change"]),
                            "final_parameter": float(r["Final Parameter"]),
                            "HDI_3pct": float(r.get("HDI 3%", np.nan)), "HDI_97pct": float(r.get("HDI 97%", np.nan)),
                            "SD": float(r.get("SD", np.nan)),
                            "inclusion_probability": float(r.get("Inclusion probability", np.nan)),
                            "credible_FDR0.2": bool(abs(float(r["Final Parameter"])) > 0),
                            "rhat": np.nan, "ess_bulk": np.nan,
                            "posterior_degenerate": False, "reliable": True,
                        })
                all_rows.extend(rows_prev)
                manifest["runs"].append({"contrast": con.name, "n_credible_primary": prev["n_credible_primary"], "resumed": True})
                continue
        try:
            rows = run_one(obs, con, out)
            if rows:
                all_rows.extend(rows)
                manifest["runs"].append({"contrast": con.name,
                                         "n_credible_primary": int(sum(r["credible_FDR0.2"] for r in rows if r["is_primary_covariate"]))})
            else:
                manifest["runs"].append({"contrast": con.name, "skipped": True})
        except Exception as e:
            import traceback; traceback.print_exc()
            manifest["runs"].append({"contrast": con.name, "error": repr(e)})

    long = pd.DataFrame(all_rows)
    long.to_csv(out / "multiome_continuous_long.csv", index=False)
    manifest["elapsed_seconds"] = time.time() - t0
    json.dump(manifest, open(out / "multiome_continuous_manifest.json", "w"), indent=2, default=str)
    _flush(f"\n[DONE] {time.time()-t0:.1f}s -> {OUTDIR}; {len(long)} rows")


if __name__ == "__main__":
    main()
