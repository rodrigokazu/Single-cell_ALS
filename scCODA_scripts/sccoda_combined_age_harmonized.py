#!/usr/bin/env python3
"""
Combined-dataset (ALS/multiome + Pineda2024/Kellis) contrast 7 only:
  (7) All donors with valid Age at death: age (standardized) as covariate.

Companion to sccoda_multiome_age_wdr49.py (the multiome-only, WDR49-resolved
version). That script deliberately excludes Pineda2024 because it has zero
WDR49+ astrocytes in every one of its 49 donors -- forcing WDR49+ into the
combined age contrast is exactly the mechanism that produced the spurious
"credible C9orf72 depletion" (PIP 0.9974) in the original, uncorrected
combined run: all 10 Pineda2024 donors with valid Age_At_Death are C9orf72
AND all 49 Pineda2024 donors are WDR49-zero, so the C9orf72 arm of that run
was partly composed of donors who structurally cannot show the WDR49+
signal.

This script sidesteps that mechanism instead of avoiding it by exclusion:
"Astrocytes WDR49" is HARMONIZED into "Astrocytes" (21 shared cell types,
matching contrasts 1-4's schema) so the tested population is one Pineda2024
can actually register a nonzero proportion for. With that fixed, the STUDY
filter is dropped and every donor with valid Age_At_Death is used -- 70
ALS/multiome donors + 10 Pineda2024 donors (all C9orf72) = 80 donors -- to
ask whether aggregate astrocyte proportion (not the WDR49+ subtype) tracks
age at death, using the largest available combined sample.

Mirrors sccoda_multiome_age_wdr49.py's lightweight obs-only load (dummy
1-column X, no 16GB counts layer) and donor-level z-scoring of continuous
covariates (feeding them raw collapsed the posterior in the pre-refit run:
SD=0, HDI width=0, inclusion prob=1 for every cell type).

Outputs -> OUTDIR/<contrast_name>/: effect_df_<cov>.csv, credible_effects.csv,
  mean_composition_by_status.csv, per_sample_composition_wide.csv,
  coda_modality.h5ad, posterior_beta.npz, contrast_manifest.json
  + OUTDIR/combined_age_harmonized_long.csv, combined_age_harmonized_manifest.json
"""
# ==============================================================================
# WHAT THIS IS
#   A sensitivity check on contrast 7, written to answer one fair objection: is the
#   age-at-death null just an artefact of excluding the external cohort?
#
# HOW IT ANSWERS THAT
#   It puts the ten external donors who have an age recorded back in, for 80 donors
#   in total, and merges WDR49+ astrocytes into astrocytes first so the population
#   being tested is one that cohort can actually register a non-zero count for.
#   That sidesteps the zero-inflation mechanism instead of avoiding it by exclusion,
#   which is what makes it a check rather than a repeat.
#
# WHAT IT PRODUCES
#   .../sccoda/reviewer_combined_age_harmonized/Combined_harmonized/All_AgeAtDeath/
#
# WHERE IT LANDS IN THE SUPPLEMENTARY TABLE
#   The Sensitivity (vii) sheet, 63 rows. SLURM job 52259919, 29 minutes.
#
# WHAT IT RETURNED
#   Nothing on age, highest inclusion probability 0.4806. The null holds with the
#   external donors included, which is the point of the exercise.
#
#   It does return two credible effects, both on the C9orf72 status term rather than
#   on age: oligodendrocytes at -0.1411 with inclusion probability 0.8662, and
#   microglia at +0.1628 with 0.7588. The job log counts only the primary covariate,
#   so it prints zero credible, and reading that as "nothing credible anywhere"
#   would be wrong. These two are not findings: in this fit all ten external donors
#   are C9orf72 while the control and sporadic arms are entirely ours, so cohort and
#   group are confounded for C9orf72 the same way they were in the artefact above.
#
#   How they entered the credible set together takes a moment to see. Microglia at 0.7588
#   cannot qualify alone, since 1 - 0.7588 = 0.2412 exceeds the 0.2 limit. Paired
#   with oligodendrocytes the set mean falls to 0.1875, which passes, so both are
#   admitted. A strong effect can carry a weaker one in under this rule.
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
# Same tqdm/JAX-Eigen race workaround as sccoda_multiome_age_wdr49.py.
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
OUTDIR  = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/sccoda/reviewer_combined_age_harmonized"
DS_LABEL = "Combined_harmonized"

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
# True on purpose. Merging WDR49+ into astrocytes is what lets the external
# donors contribute a meaningful count, which is what makes this a check.
HARMONIZE_WDR49 = True   # <-- the whole point of this variant

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
    Contrast(7, "All_AgeAtDeath",
             "All donors, age at death as covariate (combined, WDR49 harmonized)",
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
    donors_per_study = sub.groupby(STUDY_COL)[SAMPLE_COL].nunique().to_dict()
    man["donors_per_group"] = donors_per_status
    man["donors_per_study"] = donors_per_study
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
           f"({len(per_donor)} donors); donors_per_status {donors_per_status}; "
           f"donors_per_study {donors_per_study}")

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
    # NO study filter -- combined dataset, both ALS/multiome and Pineda2024/Kellis.
    _flush(f"[INFO] {DS_LABEL} (combined, no study filter): {len(obs)} cells, "
           f"{obs[SAMPLE_COL].nunique()} donors")
    _flush(f"[INFO] donors per study (pre-any-filtering): "
           f"{obs.groupby(STUDY_COL)[SAMPLE_COL].nunique().to_dict()}")

    if HARMONIZE_WDR49:
        n = int((obs[CELL_TYPE_COL] == "Astrocytes WDR49").sum())
        obs[CELL_TYPE_COL] = obs[CELL_TYPE_COL].replace({"Astrocytes WDR49": "Astrocytes"})
        _flush(f"[INFO] harmonized {n} 'Astrocytes WDR49' cells into 'Astrocytes' "
               f"(-> {obs[CELL_TYPE_COL].nunique()} shared cell types)")

    all_rows = []
    manifest = {"in_h5ad": IN_H5AD, "outdir": OUTDIR, "dataset": DS_LABEL,
                "study_filter": "none (combined ALS + Pineda2024)",
                "fdr_level": FDR_LEVEL, "harmonize_wdr49": HARMONIZE_WDR49, "rng_key": RNG_KEY,
                "n_cells": int(len(obs)), "n_donors": int(obs[SAMPLE_COL].nunique()), "runs": []}

    for con in CONTRASTS:
        man_path = out / DS_LABEL / con.name / "contrast_manifest.json"
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
    long.to_csv(out / "combined_age_harmonized_long.csv", index=False)
    manifest["elapsed_seconds"] = time.time() - t0
    json.dump(manifest, open(out / "combined_age_harmonized_manifest.json", "w"), indent=2, default=str)
    _flush(f"\n[DONE] {time.time()-t0:.1f}s -> {OUTDIR}; {len(long)} rows")


if __name__ == "__main__":
    main()
