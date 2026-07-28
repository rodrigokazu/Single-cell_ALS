#!/usr/bin/env python3
"""
Per-dataset scCODA: run the compositional contrasts SEPARATELY within each source
dataset -- our multiome cohort (study=="ALS") and the Kellis/Pineda et al. 2024
cohort (study=="Pineda2024") -- instead of only colouring a combined plot.

This directly answers "does the finding replicate in each cohort independently?"

Design
------
* Cell-type schema HARMONIZED to the 21 types shared by both cohorts: "Astrocytes
  WDR49" (a sub-annotation absent from Pineda2024) is merged back into "Astrocytes"
  so the two datasets are directly comparable. WDR49+ astrocyte composition remains
  covered (our-cohort only) by the main seven-contrast table.
* Four categorical contrasts, run inside EACH dataset (both cohorts have genotype):
    1 ALS (C9orf72+Sporadic) vs Control
    2 C9orf72-ALS vs Control
    3 Sporadic ALS vs Control
    4 C9orf72-ALS vs Sporadic ALS
  (Survival/Age contrasts are our-cohort-only -- Pineda2024 has no survival and
   age for only 10/49 donors -- so they are NOT part of this cross-dataset job.)
* FDR (discovery) threshold = 0.2, matching the paper.
* scCODA counts cells per (sample x cell type) from obs only, so we read obs alone
  and skip the expression matrix -> low memory.

Outputs (-> OUTDIR):
  <dataset>/<contrast>/ : effect_df_<cov>.csv, credible_effects.csv,
      mean_composition_by_status.csv, per_sample_composition_wide.csv,
      coda_modality.h5ad, posterior_beta.npz, contrast_manifest.json
  perdataset_all_contrasts_long.csv          (tidy: dataset x contrast x cell type)
  perdataset_comparison_wide.csv             (cell type x dataset x contrast)
  scCODA_perdataset_comparison.xlsx          (README, headline ALS-vs-Control side-by-side, ...)
  perdataset_manifest.json
"""
# ==============================================================================
# WHAT THIS IS
#   The cross-cohort check. It refits the four categorical comparisons separately
#   inside each dataset, as an independent model per cohort, so the two can be
#   neither borrows power from the other.
#
# WHAT IT PRODUCES
#   .../sccoda/reviewer_perdataset/<cohort>/<contrast>/ plus a combined long table
#   and a manifest carrying the concordance statistics.
#
# WHERE IT LANDS IN THE SUPPLEMENTARY TABLE
#   The whole Per-dataset split sheet, 168 rows, and the thirteen credible rows the
#   Credible effects sheet flags as supplementary.
#
# WHAT IT FOUND
#   Thirteen credible effects: two in ours, both microglia, and eleven in the
#   external cohort. Nothing is credible in both, in any of the four comparisons.
#   Agreement on ALS against control is weak, Spearman 0.18 across log2 fold
#   changes at p = 0.44, with the sign matching for 18 of 21 shared cell types.
#
#   State this one out loud: oligodendrocyte depletion, which is credible
#   in the pooled contrasts (i) and (iii), is credible in NEITHER cohort on its own.
#   It needs the combined sample size. That is a real result about the strength of
#   the finding, and it is better said by us than found by a reviewer.
#
# WHAT MAKES THE COMPARISON FAIR
#   We harmonise cell types to the 21 labels the datasets share, folding WDR49+
#   astrocytes into astrocytes, because the external cohort has none of them. We also
#   force the reference to the same population in both fits rather than letting
#   scCODA choose one per cohort, since effects only compare when they sit over the
#   same denominator.
#
#   The 21-type schema costs something: nothing in this analysis can say anything
#   about WDR49+ astrocytes, in either cohort.
#
# MEMORY NOTE
#   Reads obs only and hands scCODA a dummy one-column X. scCODA counts cells from
#   obs, so the 16 GB counts layer is never touched and the job stays small.
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

import gc, sys, json, time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import anndata as ad
import pertpy as pt
try:
    import arviz as az; _HAVE_ARVIZ = True
except Exception:
    _HAVE_ARVIZ = False

# --------------------------------------------------------------------------
IN_H5AD = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Final_merged_h5ad/all_cells_scvi_integrated.h5ad"
# One subdirectory per cohort per contrast, plus a long table and a manifest
# holding the cross-cohort concordance statistics.
OUTDIR  = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/sccoda/reviewer_perdataset"

CELL_TYPE_COL = "celltype_lr_full"
SAMPLE_COL    = "orig.ident"
STATUS_COL    = "type_als"
STUDY_COL     = "study"
FDR_LEVEL     = 0.2
FDR_SENSITIVITY = [0.05, 0.1, 0.2]   # credible calls recorded at each (no refit)
RNG_KEY       = 42
NUM_SAMPLES   = 10000
NUM_WARMUP    = 1000
HARMONIZE_WDR49 = True   # merge "Astrocytes WDR49" -> "Astrocytes" for comparability
# Force ONE shared reference cell type in BOTH cohorts so log2FC/credibility are on
# the same scale (scCODA effects are defined relative to the reference). Parvalbumin
# Chandelier Cells is present in both cohorts and is scCODA's automatic (low-dispersion)
# pick in the primary cohort.
REFERENCE_CELL_TYPE = "Parvalbumin Chandelier Cells"

# self-defend: pertpy/arviz create a warning-cache dir with mkdir(exist_ok=True)
# (no parents) -> ensure the parent exists regardless of how the job is launched.
os.makedirs(os.path.join(os.environ.get("HOME", "/tmp"), "arviz_data"), exist_ok=True)

DATASETS = [   # (study value, friendly label)
    ("ALS",        "OurCohort_multiome"),
    ("Pineda2024", "Pineda2024_Kellis"),
]

COLLAPSED = "type_als_collapsed_run"

@dataclass
class Contrast:
    number: int
    name: str
    label: str
    keep_statuses: list
    control_level: str
    collapse_als: bool = False

CONTRASTS = [
    Contrast(1, "ALS_vs_Control",      "ALS (C9orf72+Sporadic) vs Control", ["ALS", "Control"], "Control", collapse_als=True),
    Contrast(2, "C9orf72_vs_Control",  "C9orf72-ALS vs Control",            ["C9orf72", "Control"], "Control"),
    Contrast(3, "Sporadic_vs_Control", "Sporadic ALS vs Control",           ["Sporadic", "Control"], "Control"),
    Contrast(4, "C9orf72_vs_Sporadic", "C9orf72-ALS vs Sporadic ALS",       ["C9orf72", "Sporadic"], "Sporadic"),
]

SHORT = {"Astrocytes":"ASC","Claustrum Neurons":"CLA","Corticothalamic L6":"CT L6",
    "Endothelial Cells":"Endo","Excitatory L5":"ET L5","Interneuron L2/3":"IT L2/3",
    "Interneuron L4":"IT L4","Interneuron L5":"IT L5","Interneuron L6":"IT L6",
    "Layer 6b Neurons":"L6b","LAMP5 Interneurons":"LAMP5","Microglia":"MGC",
    "NDNF Interneurons":"NDNF","Near-Projecting Neurons":"NP","Oligodendrocytes":"ODC",
    "Oligodendrocyte Precursor Cells":"OPC","Parvalbumin Interneurons":"PVALB",
    "Parvalbumin Chandelier Cells":"PVALB ChC","Somatostatin-Expressing Neurons":"SST",
    "VIP Interneurons":"VIP","Vascular Leptomeningeal Cells":"VLMC","Astrocytes WDR49":"ASC WDR49"}


def _flush(*a, **k): print(*a, **k); sys.stdout.flush()


def _read_obs(path):
    """Read the obs frame directly (avoids loading X / the 16GB counts layer)."""
    f = h5py.File(path, "r"); og = f["obs"]
    want = [CELL_TYPE_COL, SAMPLE_COL, STATUS_COL, STUDY_COL]
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


def run_one(obs_ds, ds_label, con, outroot):
    cdir = outroot / ds_label / con.name; cdir.mkdir(parents=True, exist_ok=True)
    man = {"dataset": ds_label, "number": con.number, "name": con.name, "label": con.label,
           "fdr_level": FDR_LEVEL, "rng_key": RNG_KEY, "notes": []}

    obs = obs_ds.copy()
    obs[COLLAPSED] = obs[STATUS_COL].map({"C9orf72": "ALS", "Sporadic": "ALS", "Control": "Control"})
    status_src = COLLAPSED if con.collapse_als else STATUS_COL
    sub = obs[obs[status_src].isin(con.keep_statuses)].copy()
    # donor sanity
    ndon = sub.groupby(status_src)[SAMPLE_COL].nunique().to_dict()
    man["donors_per_group"] = ndon
    if any(v < 3 for v in ndon.values()) or len(ndon) < 2:
        man["notes"].append(f"SKIP: too few donors {ndon}")
        _flush(f"    [SKIP] {ds_label}/{con.name}: donors {ndon}")
        json.dump(man, open(cdir / "contrast_manifest.json", "w"), indent=2, default=str)
        return None

    status_key = "_cstatus"
    sub[status_key] = sub[status_src].astype(str)
    A = ad.AnnData(X=sp.csr_matrix((sub.shape[0], 1), dtype="float32"), obs=sub)

    model = pt.tl.Sccoda()
    data = model.load(A, type="cell_level", generate_sample_level=True,
                      cell_type_identifier=CELL_TYPE_COL, sample_identifier=SAMPLE_COL,
                      covariate_obs=[status_key])
    mod = f"coda_{con.name}"
    coda = data["coda"]
    m = coda.obs[status_key].astype(str).isin(con.keep_statuses).values
    data.mod[mod] = coda[m].copy()
    n_s, n_ct = data.mod[mod].n_obs, data.mod[mod].n_vars
    man["n_samples"], man["n_celltypes"] = int(n_s), int(n_ct)
    _flush(f"    {ds_label}/{con.name}: {n_s} donors x {n_ct} cell types; donors {ndon}")

    # control as reference level (treatment coding)
    o = data.mod[mod].obs
    present = sorted(o[status_key].astype(str).unique())
    cats = [con.control_level] + [x for x in con.keep_statuses if x != con.control_level and x in present]
    cats += [x for x in present if x not in cats]
    o["status_ref"] = pd.Categorical(o[status_key].astype(str), categories=cats, ordered=True)
    data.mod[mod].obs = o

    ref = REFERENCE_CELL_TYPE if REFERENCE_CELL_TYPE in list(data.mod[mod].var_names) else "automatic"
    if ref == "automatic":
        man["notes"].append(f"reference '{REFERENCE_CELL_TYPE}' absent; used automatic")
    data = model.prepare(data, modality_key=mod, formula="status_ref", reference_cell_type=ref)
    model.run_nuts(data, modality_key=mod, num_samples=NUM_SAMPLES,
                   num_warmup=NUM_WARMUP, rng_key=RNG_KEY)

    coda_fit = data[mod]
    params = coda_fit.uns.get("scCODA_params", {})
    ref_ct = params.get("reference_cell_type"); man["reference_cell_type"] = str(ref_ct)
    cov_names = [str(x) for x in np.asarray(params.get("covariate_names", [])).ravel().tolist()]
    beta = None
    try: beta = np.asarray(params["mcmc"]["samples"]["beta"])
    except Exception: pass

    # credible booleans + effect table at FDR 0.2
    try:
        cred = model.credible_effects(data, modality_key=mod, est_fdr=FDR_LEVEL)
        cred.to_csv(cdir / "credible_effects.csv")
    except Exception as e:
        man["notes"].append(f"credible_effects failed: {e!r}")
    intr, eff = model.summary_prepare(coda_fit, est_fdr=FDR_LEVEL)

    # credible sets at multiple FDR thresholds (sensitivity), no refit
    cred_by_fdr = {}
    for fdr in FDR_SENSITIVITY:
        try:
            _, e2 = model.summary_prepare(coda_fit, est_fdr=fdr)
            if isinstance(e2.index, pd.MultiIndex):
                cred_by_fdr[fdr] = {(str(c), str(ct)): (abs(float(v)) > 0)
                                    for (c, ct), v in e2["Final Parameter"].items()}
            else:
                cred_by_fdr[fdr] = {(cov_names[0] if cov_names else "effect", str(ct)): (abs(float(v)) > 0)
                                    for ct, v in e2["Final Parameter"].items()}
        except Exception as e:
            man["notes"].append(f"summary_prepare(fdr={fdr}) failed: {e!r}")

    rows = []
    per_cov = {}
    if isinstance(eff.index, pd.MultiIndex):
        for cov, s in eff.groupby(level=0):
            s = s.copy(); s.index = s.index.get_level_values(-1).astype(str); per_cov[str(cov)] = s
    else:
        per_cov[cov_names[0] if cov_names else "effect"] = eff.copy()

    for cov, df in per_cov.items():
        df.to_csv(cdir / f"effect_df_{cov}.csv")
        sd = pd.to_numeric(df.get("SD"), errors="coerce")
        lo = pd.to_numeric(df.get("HDI 3%"), errors="coerce")
        hi = pd.to_numeric(df.get("HDI 97%"), errors="coerce")
        incl = pd.to_numeric(df.get("Inclusion probability"), errors="coerce")
        width = (hi - lo).abs()
        rmap, emap = {}, {}
        if beta is not None and cov in cov_names:
            ci = cov_names.index(cov)
            if beta.ndim == 3 and ci < beta.shape[1] and beta.shape[2] == df.shape[0]:
                for j, ct in enumerate(df.index):
                    rmap[ct], emap[ct] = _split_rhat_ess(beta[:, ci, j])
        rser = pd.to_numeric(pd.Series(rmap), errors="coerce").dropna()
        frac_col = float(((sd.fillna(1).abs() < 1e-9) & (width.fillna(1) < 1e-9)).mean())
        frac_bad = float((rser > 1.1).mean()) if len(rser) else 0.0
        max_rhat = float(rser.max()) if len(rser) else np.nan
        degen = bool(frac_col >= 0.8 or frac_bad >= 0.2 or (np.isfinite(max_rhat) and max_rhat > 1.2))
        for ct in df.index:
            fp = float(pd.to_numeric(df.loc[ct, "Final Parameter"], errors="coerce"))
            rows.append({
                "dataset": ds_label, "contrast_num": con.number, "contrast": con.name,
                "contrast_label": con.label, "covariate": cov, "cell_type": ct,
                "cell_type_short": SHORT.get(ct, ct), "reference_cell_type": str(ref_ct),
                "log2_fold_change": float(pd.to_numeric(df.loc[ct, "log2-fold change"], errors="coerce")),
                "final_parameter": fp, "HDI_3pct": float(lo.get(ct, np.nan)),
                "HDI_97pct": float(hi.get(ct, np.nan)), "SD": float(sd.get(ct, np.nan)),
                "inclusion_probability": float(incl.get(ct, np.nan)),
                "credible_FDR0.2": bool(abs(fp) > 0),
                "credible_FDR0.1": bool(cred_by_fdr.get(0.1, {}).get((cov, ct), False)),
                "credible_FDR0.05": bool(cred_by_fdr.get(0.05, {}).get((cov, ct), False)),
                "rhat": float(rmap.get(ct, np.nan)), "ess_bulk": float(emap.get(ct, np.nan)),
                "posterior_degenerate": degen, "reliable": (not degen),
            })
    man["n_credible"] = int(sum(r["credible_FDR0.2"] for r in rows))

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
    _flush(f"      -> credible@FDR0.2 = {man['n_credible']} (ref={ref_ct})")

    del A, data, model, coda, coda_fit; gc.collect()
    return rows, mean


def main():
    t0 = time.time(); out = Path(OUTDIR); out.mkdir(parents=True, exist_ok=True)
    _flush(f"[INFO] reading obs from {IN_H5AD}")
    obs = _read_obs(IN_H5AD)
    _flush(f"[INFO] {len(obs)} cells; studies: {obs[STUDY_COL].value_counts().to_dict()}")

    if HARMONIZE_WDR49:
        n = int((obs[CELL_TYPE_COL] == "Astrocytes WDR49").sum())
        obs[CELL_TYPE_COL] = obs[CELL_TYPE_COL].replace({"Astrocytes WDR49": "Astrocytes"})
        _flush(f"[INFO] harmonized {n} 'Astrocytes WDR49' cells into 'Astrocytes' "
               f"(-> {obs[CELL_TYPE_COL].nunique()} shared cell types)")

    all_rows, means = [], {}
    manifest = {"in_h5ad": IN_H5AD, "outdir": OUTDIR, "fdr_level": FDR_LEVEL,
                "harmonize_wdr49": HARMONIZE_WDR49, "rng_key": RNG_KEY,
                "datasets": {}, "runs": []}

    for study_val, ds_label in DATASETS:
        obs_ds = obs[obs[STUDY_COL] == study_val].copy()
        manifest["datasets"][ds_label] = {
            "study_value": study_val, "n_cells": int(len(obs_ds)),
            "n_donors": int(obs_ds[SAMPLE_COL].nunique()),
            "n_celltypes": int(obs_ds[CELL_TYPE_COL].nunique())}
        _flush(f"\n{'='*70}\n[DATASET] {ds_label} (study={study_val}): "
               f"{len(obs_ds)} cells, {obs_ds[SAMPLE_COL].nunique()} donors\n{'='*70}")
        for con in CONTRASTS:
            try:
                res = run_one(obs_ds, ds_label, con, out)
                if res:
                    rows, mean = res
                    all_rows.extend(rows)
                    means[f"{ds_label}|{con.name}"] = mean
                    manifest["runs"].append({"dataset": ds_label, "contrast": con.name,
                                             "n_credible": int(sum(r["credible_FDR0.2"] for r in rows))})
            except Exception as e:
                import traceback; traceback.print_exc()
                manifest["runs"].append({"dataset": ds_label, "contrast": con.name, "error": repr(e)})

    long = pd.DataFrame(all_rows)
    long.to_csv(out / "perdataset_all_contrasts_long.csv", index=False)
    _flush(f"\n[INFO] long table: {len(long)} rows")

    # comparison wide: cell_type index, columns = (dataset, contrast, metric)
    blocks = []
    for ds_label in [d[1] for d in DATASETS]:
        for con in CONTRASTS:
            sub = long[(long["dataset"] == ds_label) & (long["contrast"] == con.name)]
            if len(sub) == 0: continue
            # primary covariate = the status_ref one
            covs = sub["covariate"].unique().tolist()
            pcov = next((c for c in covs if "status_ref" in c), covs[0])
            s = sub[sub["covariate"] == pcov].set_index("cell_type")
            col = f"{ds_label} | ({con.number}){con.name}"
            b = pd.DataFrame(index=s.index)
            b[(col, "log2FC")] = s["log2_fold_change"]
            b[(col, "incl.prob")] = s["inclusion_probability"]
            b[(col, "credible")] = s["credible_FDR0.2"]
            blocks.append(b)
    wide = pd.concat(blocks, axis=1) if blocks else pd.DataFrame()
    if len(wide):
        wide.columns = pd.MultiIndex.from_tuples(wide.columns)
        wide.index.name = "cell_type"
    wide.to_csv(out / "perdataset_comparison_wide.csv")

    # headline: ALS vs Control side-by-side across datasets
    head_rows = []
    c1 = long[long["contrast"] == "ALS_vs_Control"]
    cts = sorted(c1["cell_type"].unique())
    for ct in cts:
        row = {"cell_type": ct, "short": SHORT.get(ct, ct)}
        for ds_label in [d[1] for d in DATASETS]:
            r = c1[(c1["dataset"] == ds_label) & (c1["cell_type"] == ct)]
            if len(r):
                r = r.iloc[0]
                row[f"{ds_label}_log2FC"] = round(r["log2_fold_change"], 3)
                row[f"{ds_label}_inclProb"] = round(r["inclusion_probability"], 3)
                row[f"{ds_label}_credible"] = r["credible_FDR0.2"]
            else:
                row[f"{ds_label}_log2FC"] = np.nan
                row[f"{ds_label}_inclProb"] = np.nan
                row[f"{ds_label}_credible"] = None
        head_rows.append(row)
    head = pd.DataFrame(head_rows)

    # ---- cross-dataset concordance for ALS vs Control ----
    concordance = {}
    try:
        from scipy.stats import spearmanr
        labels = [d[1] for d in DATASETS]
        piv = (long[long["contrast"] == "ALS_vs_Control"]
               .pivot_table(index="cell_type", columns="dataset", values="log2_fold_change"))
        if all(l in piv.columns for l in labels):
            common = piv[labels].dropna()
            a, b = common[labels[0]].values, common[labels[1]].values
            rho, p = spearmanr(a, b) if len(common) > 2 else (np.nan, np.nan)
            sign_agree = float((np.sign(a) == np.sign(b)).mean()) if len(common) else np.nan
            c1 = long[long["contrast"] == "ALS_vs_Control"]
            cred_a = set(c1[(c1["dataset"] == labels[0]) & (c1["credible_FDR0.2"])]["cell_type"])
            cred_b = set(c1[(c1["dataset"] == labels[1]) & (c1["credible_FDR0.2"])]["cell_type"])
            concordance = {
                "contrast": "ALS_vs_Control",
                "n_shared_cell_types": int(len(common)),
                "spearman_log2FC": None if not np.isfinite(rho) else round(float(rho), 3),
                "spearman_p": None if not np.isfinite(p) else round(float(p), 4),
                "sign_agreement_frac": None if not np.isfinite(sign_agree) else round(sign_agree, 3),
                f"credible_{labels[0]}": sorted(cred_a),
                f"credible_{labels[1]}": sorted(cred_b),
                "credible_in_both": sorted(cred_a & cred_b),
            }
            _flush(f"\n[CONCORDANCE ALS_vs_Control] spearman(log2FC)={concordance['spearman_log2FC']} "
                   f"sign-agree={concordance['sign_agreement_frac']} "
                   f"credible_both={concordance['credible_in_both']}")
    except Exception as e:
        concordance = {"error": repr(e)}

    readme = pd.DataFrame({"Per-dataset scCODA comparison": [
        "Compositional contrasts run SEPARATELY within each source cohort.",
        f"FDR (discovery) threshold = {FDR_LEVEL}. Fixed seed = {RNG_KEY}.",
        "Datasets: OurCohort_multiome (study=ALS) and Pineda2024_Kellis (study=Pineda2024).",
        "Cell-type schema harmonized to the 21 types shared by both cohorts:",
        "  'Astrocytes WDR49' merged into 'Astrocytes' (Pineda2024 lacks this sub-annotation).",
        "Contrasts (each run inside each dataset): (1) ALS vs Control, (2) C9orf72 vs Control,",
        "  (3) Sporadic vs Control, (4) C9orf72 vs Sporadic.",
        "Survival/Age contrasts are our-cohort only (Pineda2024 has no survival) -> see the",
        "  main seven-contrast table, not this file.",
        "A cell type is credible when Final Parameter != 0 (inclusion prob >= model threshold).",
        f"Shared reference cell type forced in BOTH cohorts: '{REFERENCE_CELL_TYPE}' (scCODA effects are",
        "  relative to a reference; a common reference makes log2FC/credibility comparable across cohorts).",
        f"Credible calls recorded at FDR {FDR_SENSITIVITY} (sensitivity); primary threshold 0.2.",
        "Concordance sheet: Spearman correlation of log2FC + sign agreement + shared credible sets.",
        "Sheets: README | ALS_vs_Control_by_dataset (headline replication) | Concordance |",
        "  Comparison_wide | All_contrasts_long | Mean_composition | Diagnostics.",
    ]})

    diag = (long.groupby(["dataset", "contrast_num", "contrast", "covariate"])
            .agg(n_cell_types=("cell_type", "nunique"),
                 n_credible=("credible_FDR0.2", "sum"),
                 max_rhat=("rhat", "max"), min_ess=("ess_bulk", "min"),
                 degenerate=("posterior_degenerate", "any"),
                 reference_cell_type=("reference_cell_type", "first"))
            .reset_index()) if len(long) else pd.DataFrame()

    xlsx = out / "scCODA_perdataset_comparison.xlsx"
    engine = None
    for e in ("openpyxl", "xlsxwriter"):
        try:
            __import__(e); engine = e; break
        except Exception:
            continue
    if engine:
        with pd.ExcelWriter(xlsx, engine=engine) as xw:
            readme.to_excel(xw, "README", index=False)
            head.to_excel(xw, "ALS_vs_Control_by_dataset", index=False)
            if concordance:
                pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v)
                               for k, v in concordance.items()}]).T.rename(
                    columns={0: "value"}).to_excel(xw, "Concordance")
            if len(wide): wide.to_excel(xw, "Comparison_wide")
            long.to_excel(xw, "All_contrasts_long", index=False)
            mc = []
            for k, m in means.items():
                mm = m.copy(); mm.insert(0, "dataset_contrast", k); mm.index.name = "status"
                mc.append(mm.reset_index())
            if mc: pd.concat(mc, ignore_index=True).to_excel(xw, "Mean_composition", index=False)
            if len(diag): diag.to_excel(xw, "Diagnostics", index=False)
        _flush(f"[INFO] wrote {xlsx}")
    else:
        _flush("[WARN] no xlsx engine; CSVs only")

    manifest["concordance_ALS_vs_Control"] = concordance
    manifest["reference_cell_type"] = REFERENCE_CELL_TYPE
    manifest["fdr_sensitivity"] = FDR_SENSITIVITY
    manifest["elapsed_seconds"] = time.time() - t0
    json.dump(manifest, open(out / "perdataset_manifest.json", "w"), indent=2, default=str)
    _flush(f"\n[DONE] {time.time()-t0:.1f}s -> {OUTDIR}")


if __name__ == "__main__":
    main()
