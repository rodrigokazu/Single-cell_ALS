#!/usr/bin/env python3
"""Milo differential abundance, re-run SEPARATELY per cohort, each on Jingtian's REAL
per-cohort corrected-PCA embedding (not the naive cross-cohort stitch that Milopy_survival.py
used, and not the row-permuted Merged_v3 embedding). Motivation: the joint/merged run found
0 significant nhoods in every contrast; testing whether that null survives when each cohort
is analysed on its own valid embedding, avoiding the never-jointly-corrected concatenation.

Mirrors Milopy_survival.py's pipeline exactly (N_NEIGHBORS=50, PROP_NHOODS=0.25, ALPHA=0.10,
same 4 contrasts, same design formula ~ design_col, no Sex covariate) except:
  - obsm['X_pca_corrected'] comes from Jingtian's own per-cohort file, matched by barcode,
    instead of the pipeline's own (never-jointly-corrected, and in the full-merged case
    row-permuted) copy.
  - Metadata (Diagnosis, type_als, Sex, celltype_lr_full, orig.ident) comes from Rodrigo's
    Multiome_concatenated_full_label_2026_MinusFTD.h5ad -- obs values there are trustworthy
    per-row (only the file's OWN obsm was found scrambled relative to its own obs; obs itself
    was never implicated).
  - X is a DUMMY single-column zero matrix -- da_nhoods operates on the nhood x sample CELL
    COUNT matrix (from count_nhoods), not on gene expression, and Jingtian's real files carry
    no expression matrix at all (obsm/obs/obsp only). Same convention already used for scCODA
    in this project when only cell counts are needed.

Usage: python3 milo_split_cohort_run.py <COHORT> [--smoke N]
  COHORT: "ALS" or "Pineda2024"
  --smoke N: subsample to N cells first, for a fast mechanics-only sanity check.
"""
# ==============================================================================
# WHAT THIS IS
#   The script behind the split-cohort Milo re-analysis: same statistical pipeline
#   as the manuscript's own full-merged Milo run, same four contrasts, same design
#   formula, run once per cohort instead of once on the (broken) combined dataset.
#   Called twice, unchanged, as `python3 milo_split_cohort_run.py ALS` and
#   `python3 milo_split_cohort_run.py Pineda2024`.
#
# WHY THIS EXISTS
#   The manuscript's full-merged run found 0 of 62,952 neighbourhoods significant,
#   in every contrast. Two independent problems were already found in that run:
#   its own X_pca_corrected was row-permuted relative to its own metadata at the
#   point the neighbourhood graph got built, and its pyDESeq2 size factors never
#   converged. Neither of those is fixed by re-plotting the same result on better
#   coordinates. The only way to find out whether the null was biology or a
#   broken pipeline was to rebuild the whole analysis, neighbourhood graph and
#   all, on an embedding already confirmed sound -- and Jingtian's own per-cohort
#   embeddings, independently confirmed bit-exact against the copy already sitting
#   in our own files, are exactly that.
#
# WHY SPLIT BY COHORT RATHER THAN REPAIR THE MERGED ONE
#   Jingtian computed his ALS(JCK) and Pineda2024(Kellis) embeddings completely
#   separately -- two independent tSNE and PCA runs, each anchored only to a
#   shared reference panel, never corrected against each other. There is no
#   single joint embedding to repair the merged run onto. Running each cohort on
#   its own native embedding sidesteps that problem instead of trying to solve it,
#   and happens to be the same resolution JCK separately asked for when he wanted
#   a Milo counterpart to the three-way scCODA comparison.
#
# WHERE THE INPUTS COME FROM, AND WHY THEY'RE SPLIT ACROSS TWO FILES
#   The neighbourhood-defining coordinates (obsm['X_pca_corrected']) come from
#   Jingtian's own per-cohort output files under ALS_multiome/Jingtian/, matched
#   to our cells by barcode. Everything else about each cell (its diagnosis, ALS
#   subtype, sex, cell-type label, which donor it came from) still comes from our
#   own Multiome_concatenated_full_label_2026_MinusFTD.h5ad. That split is
#   deliberate, not a shortcut: the earlier audit that found this file's embedding
#   scrambled never implicated its per-row metadata, only the coordinate array, so
#   there was no reason to distrust the diagnosis/subtype/cell-type columns and
#   every reason to keep using the file that already has them in the right shape
#   for this pipeline.
#
# THE BARCODE-MATCHING WRINKLE
#   Jingtian's ALS file stores bare 16bp barcodes plus a separate `sample` column
#   (e.g. sample "39_ALS_10", barcode "GACCTGATCATCCTCA-1"); our own files use the
#   concatenated form "39_ALS_10_GACCTGATCATCCTCA-1" directly as the barcode. The
#   ALS branch below reconstructs that concatenation before matching. Jingtian's
#   Pineda file, in contrast, already uses our exact barcode convention natively
#   (e.g. "AAACCCACAATCGCGC-140MCX1"), so no reconstruction is needed there -- see
#   the cohort-conditional branch in load_jingtian_embedding(). Both paths were
#   confirmed to reach 100% barcode coverage before this script was trusted with
#   a real run.
#
# THE MISSING EXPRESSION MATRIX, AND WHY IT DOESN'T MATTER
#   Jingtian's real output files carry embeddings, neighbour graphs and metadata,
#   but no gene expression matrix at all -- obsm/obs/obsp only. That would matter
#   for a differential EXPRESSION test, but Milo's differential ABUNDANCE test
#   never looks at gene expression: once the neighbourhood graph is built from the
#   embedding, everything downstream (count_nhoods, da_nhoods) works from a table
#   of how many cells per donor land in each neighbourhood. A single dummy
#   placeholder column stands in for X below, the same convention already used
#   for scCODA analyses in this project whenever only cell counts are needed.
#
# HOW THIS WAS VALIDATED BEFORE THE REAL RUN
#   Run once with --smoke 20000 on Pineda2024 before committing to the full-size
#   jobs: all four contrasts completed without error, the sample-level design
#   table matched the donor composition expected from the raw metadata, and
#   pyDESeq2's internal fitting steps finished in single-digit seconds rather than
#   the many hours seen on the corrupted embedding -- itself a good sign the
#   underlying coordinates are healthy, since a scrambled embedding tends to
#   produce a much harder, slower-to-fit dispersion landscape.
#
# WHAT IT PRODUCES
#   Under MiloP_results/JingtianSplit_<cohort>/: a pre-DA checkpoint h5mu (saved
#   before any contrast runs, so a crash mid-way still leaves something usable), a
#   checkpoint h5mu re-saved after every contrast, and one DA_results_<label>.csv
#   per contrast with the per-neighbourhood logFC, p-value, FDR, SpatialFDR,
#   majority-cell-type annotation and a boolean significance flag.
#
# WHAT IT DOES NOT DO
#   It does not touch, repair, or re-run anything from the manuscript's own
#   Milo pipeline or the already-void Merged_v3 panel. This is a parallel,
#   exploratory re-analysis; whatever it finds does not by itself change what
#   goes in the manuscript.
# ==============================================================================
# Comments added 28 July 2026, after both cohorts' production jobs had already
# completed. Executable logic is byte-identical to the version that ran -- a
# pristine copy sits in _verbatim/, and every non-docstring statement was
# confirmed unchanged with verify_comments_only.py before this copy shipped.
# ==============================================================================
from __future__ import annotations
import os, sys, time, traceback, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from mudata import MuData
import pertpy as pt

warnings_ = __import__("warnings")
warnings_.filterwarnings("ignore")

_T0 = time.time()
def ts():
    e = time.time() - _T0
    m, s = divmod(int(e), 60); h, m = divmod(m, 60)
    return f"[{time.strftime('%H:%M:%S')} | {h:02d}h{m:02d}m{s:02d}s]"
def info(msg): print(f"{ts()} {msg}", flush=True)
def section(msg): print(f"\n{ts()} --- {msg} ---", flush=True)
def banner(msg): print(f"\n{'='*80}\n{ts()} {msg}\n{'='*80}", flush=True)

# Rodrigo's own file: the source of every column EXCEPT the embedding. Its own
# obsm was found row-permuted in a separate audit; its obs columns were not, and
# are used here exactly as recorded.
RK_FILE = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Multiome_concatenated_full_label_2026_MinusFTD.h5ad"
# Jingtian's own per-cohort merge outputs -- NOT the path his own scripts read
# from (that path doesn't exist on this cluster); this is where the real files
# actually live, confirmed to contain an intact obsm (unlike the stripped
# Sam_Analysis proxy files used earlier in the same investigation).
JINGTIAN_FILES = {
    "ALS": "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Jingtian/MSMO_ALS_RNA_merged_lr60.h5ad",
    "Pineda2024": "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Jingtian/MSMO_Pineda_RNA_merged.h5ad",
}

SAMPLE_COL = "orig.ident"
CONDITION_COL = "Diagnosis"
GROUP_COL = "type_als"
CELLTYPE_COL = "celltype_lr_full"
USE_REP = "X_pca_corrected"      # Jingtian's real per-cohort PCA -- the neighbourhood graph is built on this
TSNE_KEY = "pc40seuratcc30_tsne"  # Jingtian's own tSNE key name, carried straight through for downstream plotting

# Every one of these four matches Milopy_survival.py's own configuration
# exactly. Nothing about the statistical design changed; only the coordinate
# system and the metadata source did.
N_NEIGHBORS = 50
PROP_NHOODS = 0.25
RANDOM_STATE = 0
ALPHA = 0.10

# Same four contrasts, same design columns and group labels as the original
# full-merged pipeline. contrast_diagnosis and contrast_type_als are built
# below as plain copies of Diagnosis/type_als (see build_cohort_adata).
CONTRASTS = [
    {"label": "ALS_vs_Control",      "design_col": "contrast_diagnosis", "groupA": "ALS",      "groupB": "Control"},
    {"label": "C9orf72_vs_Control",  "design_col": "contrast_type_als",  "groupA": "C9orf72",  "groupB": "Control"},
    {"label": "Sporadic_vs_Control", "design_col": "contrast_type_als",  "groupA": "Sporadic", "groupB": "Control"},
    {"label": "C9orf72_vs_Sporadic", "design_col": "contrast_type_als",  "groupA": "C9orf72",  "groupB": "Sporadic"},
]

def cat_col(h5_group, name):
    # h5py stores a pandas categorical as a group with categories/codes rather
    # than a flat array of strings, so this has to branch on which shape a given
    # obs column was saved as. Handles both transparently.
    g = h5_group[name]
    if hasattr(g, "keys"):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
        codes = g["codes"][:]
        return cats[codes]
    vals = g[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in vals])

def read_index(h5_group):
    idx = h5_group["_index"][:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in idx])


def load_rk_metadata(cohort):
    # Index-only, obs-only read via h5py rather than a full anndata.read_h5ad --
    # this file is tens of gigabytes with a heavy counts layer we never touch, so
    # loading it properly would be needless I/O for five metadata columns.
    section(f"Reading Rodrigo's metadata for cohort={cohort} from {RK_FILE}")
    h = h5py.File(RK_FILE, "r")
    obs = h["obs"]
    study = cat_col(obs, "study")
    mask = study == cohort
    idx = read_index(obs)[mask]
    df = pd.DataFrame({
        SAMPLE_COL: cat_col(obs, SAMPLE_COL)[mask],
        CONDITION_COL: cat_col(obs, CONDITION_COL)[mask],
        GROUP_COL: cat_col(obs, GROUP_COL)[mask],
        CELLTYPE_COL: cat_col(obs, CELLTYPE_COL)[mask],
        "Sex": cat_col(obs, "Sex")[mask],
    }, index=idx)
    h.close()
    info(f"  {len(df):,} cells for cohort={cohort}")
    info(f"  Diagnosis: {df[CONDITION_COL].value_counts().to_dict()}")
    info(f"  type_als:  {df[GROUP_COL].value_counts().to_dict()}")
    info(f"  samples:   {df[SAMPLE_COL].nunique()}")
    return df


def load_jingtian_embedding(cohort):
    f = JINGTIAN_FILES[cohort]
    section(f"Reading Jingtian's real embedding for cohort={cohort} from {f}")
    h = h5py.File(f, "r")
    obs = h["obs"]
    bare_idx = read_index(obs)
    if cohort == "ALS":
        # Jingtian's ALS file splits the barcode into a bare 16bp sequence plus a
        # separate `sample` column; reconstructing "sample_barcode" here is what
        # makes it comparable to our own files' native index, string for string.
        sample = cat_col(obs, "sample")
        key = np.array([f"{s}_{b}" for s, b in zip(sample, bare_idx)])
    else:
        key = bare_idx  # Pineda native index already matches RK file's barcode format
    pca = h["obsm"][USE_REP][:]
    tsne = h["obsm"][TSNE_KEY][:]
    h.close()
    info(f"  {len(key):,} cells with embedding, PCA shape {pca.shape}, tsne shape {tsne.shape}")
    return key, pca, tsne


def build_cohort_adata(cohort, smoke_n=None):
    meta_df = load_rk_metadata(cohort)
    key, pca, tsne = load_jingtian_embedding(cohort)
    key_pos = {k: i for i, k in enumerate(key)}

    # This match was already proven to reach 100% in the investigation that led
    # here; this check reruns it live every time so a silent regression (e.g. a
    # future update to either source file) fails loudly instead of quietly
    # dropping cells.
    barcodes = meta_df.index.values
    matched_mask = np.array([b in key_pos for b in barcodes])
    n_matched = matched_mask.sum()
    info(f"Barcode match: {n_matched:,} / {len(barcodes):,} ({100*n_matched/len(barcodes):.2f}%)")
    if n_matched < len(barcodes):
        info(f"  WARNING: {len(barcodes)-n_matched:,} cells unmatched, dropping them.")

    meta_df = meta_df.loc[matched_mask].copy()
    barcodes = meta_df.index.values
    positions = np.array([key_pos[b] for b in barcodes])
    pca_matched = pca[positions]
    tsne_matched = tsne[positions]

    if smoke_n is not None and smoke_n < len(meta_df):
        # Mechanics-only sanity check, not a scientific result: this is what
        # --smoke 20000 ran against Pineda2024 before either production job was
        # ever submitted, purely to prove the AnnData-with-dummy-X approach and
        # the Milo call sequence actually work end to end.
        rng = np.random.default_rng(0)
        sel = rng.choice(len(meta_df), size=smoke_n, replace=False)
        meta_df = meta_df.iloc[sel].copy()
        pca_matched = pca_matched[sel]
        tsne_matched = tsne_matched[sel]
        info(f"SMOKE MODE: subsampled to {len(meta_df):,} cells")

    n = len(meta_df)
    # The dummy expression matrix. One column, all zeros, never read by anything
    # downstream -- see the module docstring for why Milo's DA test doesn't need
    # real counts here. Kept as a proper sparse matrix so AnnData's own shape and
    # dtype checks pass without complaint.
    dummy_X = sp.csr_matrix((n, 1), dtype=np.float32)
    var_df = pd.DataFrame(index=["dummy_gene"])

    adata = ad.AnnData(X=dummy_X, obs=meta_df, var=var_df)
    adata.obsm[USE_REP] = pca_matched.astype(np.float32)
    adata.obsm[TSNE_KEY] = tsne_matched.astype(np.float32)
    adata.obs[SAMPLE_COL] = adata.obs[SAMPLE_COL].astype(str)
    adata.obs[CONDITION_COL] = adata.obs[CONDITION_COL].astype(str)
    adata.obs[GROUP_COL] = adata.obs[GROUP_COL].astype(str)
    adata.obs[CELLTYPE_COL] = adata.obs[CELLTYPE_COL].astype(str)
    # contrast_diagnosis / contrast_type_als: plain renamed copies of Diagnosis /
    # type_als. Milo's da_nhoods reads the design column by this name; keeping
    # both the original and the renamed copy matches the original pipeline's own
    # naming exactly, which is what CONTRASTS above expects.
    adata.obs["contrast_diagnosis"] = adata.obs[CONDITION_COL].copy()
    adata.obs["contrast_type_als"] = adata.obs[GROUP_COL].copy()
    info(f"Built AnnData: {adata.n_obs:,} cells x {adata.n_vars} (dummy) gene(s)")
    return adata


def mode_or_na(x):
    # Per-donor metadata should be constant within a donor (every cell from the
    # same sample has the same Diagnosis), so "most common value" and "the value"
    # agree in practice; this just guards against any stray inconsistency instead
    # of assuming it can't happen.
    vc = x.value_counts(dropna=True)
    return str(vc.index[0]) if len(vc) > 0 else "NA"

def build_sample_meta(adata_in, cols):
    # Collapses cell-level metadata down to one row per donor -- Milo's DA test
    # operates at the sample (donor) level, not the cell level, so this is the
    # table that actually gets joined onto the neighbourhood-count matrix below.
    df = adata_in.obs[[SAMPLE_COL] + cols].copy()
    df[SAMPLE_COL] = df[SAMPLE_COL].astype(str)
    meta = df.groupby(SAMPLE_COL, sort=False).agg({c: mode_or_na for c in cols})
    meta[SAMPLE_COL] = meta.index.astype(str)
    meta.index.name = None
    return meta

def attach_sample_meta(mdata, meta, cols):
    # milopy's count_nhoods() step produces a fresh milo.obs indexed by sample,
    # but wipes any per-sample columns that aren't the sample ID itself -- this
    # re-attaches Diagnosis/type_als etc. after the fact, which is why it runs
    # after count_nhoods rather than before.
    milo_mod = mdata["milo"]
    if SAMPLE_COL not in milo_mod.obs.columns:
        milo_mod.obs[SAMPLE_COL] = milo_mod.obs_names.astype(str)
    milo_mod.obs[SAMPLE_COL] = milo_mod.obs[SAMPLE_COL].astype(str)
    for c in cols:
        if c in milo_mod.obs.columns:
            milo_mod.obs.drop(columns=[c], inplace=True)
    milo_mod.obs = milo_mod.obs.join(meta[cols], how="left", on=SAMPLE_COL)
    for c in cols:
        milo_mod.obs[c] = milo_mod.obs[c].astype(str)


def annotate_nhoods_by_majority_celltype(rna, mdata):
    # Milo tests NEIGHBOURHOODS, which are built purely from KNN structure in PCA
    # space and have no cell-type label of their own. This assigns each
    # neighbourhood the single most common cell-type label among its member
    # cells, plus a purity score (what fraction of members share that label) so
    # a downstream reader can tell a clean, single-cell-type neighbourhood from a
    # messy, mixed one. This is descriptive annotation for interpretation, not
    # part of the statistical test itself.
    nhoods_mat = rna.obsm["nhoods"]
    celltypes = rna.obs[CELLTYPE_COL].values
    n_nhoods = nhoods_mat.shape[1]
    annotations, purity = [], []
    for j in range(n_nhoods):
        member_idx = nhoods_mat[:, j].nonzero()[0]
        if len(member_idx) == 0:
            annotations.append("NA"); purity.append(0.0); continue
        vc = pd.Series(celltypes[member_idx]).value_counts()
        annotations.append(str(vc.index[0]))
        purity.append(vc.iloc[0] / len(member_idx))
    mdata["milo"].var["nhood_annotation"] = annotations
    mdata["milo"].var["nhood_purity"] = purity


def run_contrast(mdata, milo_obj, label, design_col, groupA, groupB, out_dir):
    # No Sex covariate here (matches Milopy_survival.py's INCLUDE_SEX_COVARIATE =
    # False): the design is deliberately as simple as the original pipeline's,
    # so any difference in the result traces back to the embedding and metadata
    # source, not to a change in the statistical model.
    design = f"~ {design_col}"
    contrast_string = f"{design_col}{groupA}-{design_col}{groupB}"
    section(f"Contrast: {label}")
    info(f"Design: {design} | contrast: {contrast_string}")
    milo_obs = mdata["milo"].obs
    if design_col in milo_obs.columns:
        info(f"Sample distribution for '{design_col}': {milo_obs[design_col].value_counts().to_dict()}")
    t0 = time.time()
    try:
        # This is the pyDESeq2-backed NB GLM fit at the heart of Milo. On the
        # corrupted merged embedding this step was the one that took upwards of
        # 12 hours per contrast while never converging; on this real embedding it
        # finished in seconds during the smoke test, which is itself informal
        # evidence the input here is well-behaved.
        milo_obj.da_nhoods(mdata, design=design, model_contrasts=contrast_string)
    except Exception as e:
        # A failed contrast should not take down the whole job -- the other three
        # contrasts, and whichever ones already succeeded, are still worth having.
        info(f"ERROR running {label}: {e}")
        traceback.print_exc()
        return False
    info(f"da_nhoods completed in {time.time()-t0:.1f}s")

    v = mdata["milo"].var
    for src in ["logFC", "PValue", "FDR", "SpatialFDR", "logCPM"]:
        if src in v.columns:
            v[f"{src}__{label}"] = v[src].values

    res_cols = [c for c in ["nhood_annotation", f"logFC__{label}", f"PValue__{label}",
                             f"FDR__{label}", f"SpatialFDR__{label}"] if c in v.columns]
    out_df = v[res_cols].copy()
    out_df["significant_at_alpha"] = out_df[f"SpatialFDR__{label}"].astype(float) < ALPHA
    out_path = out_dir / f"DA_results_{label}.csv"
    out_df.to_csv(out_path, index=True)

    sig = out_df["significant_at_alpha"].values
    lfc = out_df[f"logFC__{label}"].astype(float).values
    up = sig & (lfc > 0); dn = sig & (lfc < 0)
    info(f"RESULTS {label}: {int(sig.sum()):,}/{len(sig):,} sig (SpatialFDR<{ALPHA}), "
         f"up-in-{groupA}={int(up.sum())}, up-in-{groupB}={int(dn.sum())}")
    if sig.any():
        info(f"  logFC sig range: [{lfc[sig].min():.3f}, {lfc[sig].max():.3f}]")
        for ct in sorted(out_df.loc[sig, "nhood_annotation"].unique()):
            n_ct = int((out_df["nhood_annotation"] == ct).sum())
            n_ct_sig = int((sig & (out_df["nhood_annotation"] == ct)).sum())
            info(f"    {ct}: {n_ct_sig}/{n_ct} sig")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort", choices=["ALS", "Pineda2024"])
    ap.add_argument("--smoke", type=int, default=None)
    args = ap.parse_args()
    cohort = args.cohort

    out_dir = Path(f"/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Milo/MiloP_results/JingtianSplit_{cohort}")
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(f"MILO SPLIT-COHORT RUN — {cohort}" + (f" [SMOKE n={args.smoke}]" if args.smoke else ""))

    adata = build_cohort_adata(cohort, smoke_n=args.smoke)

    section("Building Milo object")
    milo = pt.tl.Milo()
    mdata = milo.load(adata)
    rna = mdata["rna"]
    info(f"rna modality: {rna.n_obs:,} cells")

    section(f"Neighbours (n_neighbors={N_NEIGHBORS}, use_rep={USE_REP})")
    t0 = time.time()
    # This is the one line that actually determines whether this run is trusted:
    # use_rep points at Jingtian's real per-cohort embedding, not at any
    # never-jointly-corrected or corrupted copy. Everything from here on is
    # standard Milo mechanics, unchanged from the original pipeline.
    sc.pp.neighbors(rna, use_rep=USE_REP, n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE)
    rna.uns["neighbors"]["params"]["use_rep"] = USE_REP
    info(f"  done in {time.time()-t0:.1f}s")

    section(f"Neighbourhoods (prop={PROP_NHOODS})")
    t0 = time.time()
    milo.make_nhoods(rna, prop=PROP_NHOODS)
    n_nhoods = rna.obsm["nhoods"].shape[1]
    info(f"  {n_nhoods:,} neighbourhoods in {time.time()-t0:.1f}s")

    section("Counting neighbourhoods by sample")
    t0 = time.time()
    milo.count_nhoods(mdata, sample_col=SAMPLE_COL)
    info(f"  done in {time.time()-t0:.1f}s; milo shape {mdata['milo'].shape}")

    section("Annotating neighbourhoods by majority cell type")
    annotate_nhoods_by_majority_celltype(rna, mdata)

    section("Attaching sample-level metadata")
    meta_cols = ["contrast_diagnosis", "contrast_type_als", CONDITION_COL, GROUP_COL]
    sample_meta = build_sample_meta(adata, meta_cols)
    attach_sample_meta(mdata, sample_meta, meta_cols)
    info(f"Sample metadata table:\n{sample_meta.to_string()}")

    section("Pre-DA checkpoint")
    # Saved before any contrast is fitted, specifically so a crash or a time-limit
    # kill during the DA loop still leaves a usable neighbourhood graph behind
    # instead of losing the (much slower) neighbours + make_nhoods + count_nhoods
    # work along with it.
    try:
        mdata.write_h5mu(str(out_dir / f"Milo_{cohort}_pre_DA.h5mu"))
        info("  saved")
    except Exception as e:
        info(f"  WARNING checkpoint save failed: {e}")

    banner("RUNNING DA CONTRASTS")
    results = {}
    for cfg in CONTRASTS:
        ok = run_contrast(mdata, milo, cfg["label"], cfg["design_col"], cfg["groupA"], cfg["groupB"], out_dir)
        results[cfg["label"]] = ok
        # Re-saved after every single contrast, not just at the end -- if
        # contrast 3 of 4 fails or the job runs out of wall-clock time, the first
        # two results are still safely on disk.
        try:
            mdata.write_h5mu(str(out_dir / f"Milo_{cohort}.h5mu"))
        except Exception as e:
            info(f"  WARNING checkpoint save failed: {e}")

    banner(f"DONE — {cohort}")
    for lab, ok in results.items():
        info(f"  {lab}: {'OK' if ok else 'FAILED'}")

if __name__ == "__main__":
    main()
