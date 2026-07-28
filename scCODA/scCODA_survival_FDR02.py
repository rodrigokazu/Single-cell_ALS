#!/usr/bin/env python3
"""
Run pertpy scCODA for multiple contrasts at FDR 0.2 (discovery threshold,
as used in the scCODA paper for all real-data analyses).

Output directory: …/merged_dataset_multi_contrast_v2_FDR02/
Subdirectories per contrast: <contrast_name>/

Contrasts:
  1) ALS (collapsed C9orf72 + Sporadic) vs Control
  2) C9orf72 vs Control
  3) Sporadic vs Control
  4) C9orf72 vs Sporadic
  5) C9orf72 vs Sporadic (adjusted for Survival)
  6) ALS samples only — Survival as sole covariate
  7) All samples — Age at Death as covariate

Cell type column: celltype_lr_full
Source status column: adata.obs["type_als"]
"""
# ==============================================================================
# WHAT THIS IS
#   The canonical run behind the paper's compositional analysis. Everything the
#   manuscript says about cell-type proportions traces back to this script, and
#   Figure 1F is its ALS_vs_Control contrast.
#
# WHAT IT PRODUCES
#   .../sccoda/merged_dataset_multi_contrast_v2_FDR02/<contrast>/ , seven
#   subdirectories, one per contrast. Each holds the fitted effect tables, the
#   credible-effect list, per-sample and mean compositions, and a postrun .h5mu.
#
# WHERE IT LANDS IN THE SUPPLEMENTARY TABLE
#   Contrasts (i) to (iv) in Supplementary_Table_scCODA_seven_contrasts_FINAL.xlsx
#   come from here and are authoritative. Its contrasts 5, 6 and 7 do NOT: see the
#   warning below.
#
# HOW IT WAS RUN
#   sbatch run_sccoda.sh, partition batch, account mpsnyder, conda env pertpy_env.
#   Dated 10 March 2026, which is months before the review arrived, so this is
#   genuinely the pre-submission analysis rather than anything rebuilt afterwards.
#
# BEFORE REUSING THIS SCRIPT
#   Contrasts 5, 6 and 7 feed their continuous covariates in RAW. Look for
#   formula="status_ref + Survival" further down: no standardisation. That collapsed
#   the posterior. Every cell type came back with a standard deviation of zero, a
#   zero-width interval and an inclusion probability of exactly 1.00, with r-hat
#   between 2.3 and 2.7. It reads like 21 of 22 cell types being credible and it is
#   a fitting failure. sccoda_refit_scaled_c6_c7.py replaced those three by
#   z-scoring the covariates first, and contrast 7 was replaced again by
#   sccoda_multiome_age_wdr49.py for a separate reason. Contrasts 1 to 4 are
#   categorical, never touch that code path, and are fine.
#
# A DISCREPANCY TO CARRY
#   The manuscript reports the oligodendrocyte result as inclusion probability 0.92
#   with beta -0.14. This run gives 0.9310 and -0.1551, and so does every other run
#   on disk. The mean proportions it produces, 31.4 percent falling to 27.4 percent,
#   are what the manuscript quotes, which is how we confirmed this is the Figure 1F
#   source. Correct the text against the table, not the other way round.
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
os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)

from pathlib import Path
import json
import pickle
import sys
import time
import platform
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scanpy as sc
import mudata as md
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list

import pertpy as pt


# =====================================================================
# User-configurable parameters
# =====================================================================
IN_H5AD = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Final_merged_h5ad/all_cells_scvi_integrated.h5ad"
# Output root. The _v2_FDR02 suffix marks this as the FDR 0.2 pass; an earlier
# _v2 directory holds the same contrasts at pertpy's default 0.05.
OUTDIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/sccoda/merged_dataset_multi_contrast_v2_FDR02"

CELL_TYPE_COL = "celltype_lr_full"
SAMPLE_COL = "orig.ident"

STATUS_SOURCE_COL = "type_als"
SURVIVAL_COL = "Survival"
AGE_COL = "Age_At_Death"

# =====================================================================
# FDR threshold — the only change from the FDR 0.05 run
# =====================================================================
# The discovery threshold used for every real-data analysis in the original
# scCODA paper. We apply it throughout and describe it that way in the response.
# Not a fixed cut on inclusion probability: scCODA ranks cell types and keeps
# the largest set whose mean (1 - inclusion probability) stays under this.
FDR_LEVEL = 0.2


# =====================================================================
# Cell-type colour palette  (keyed by celltype_lr_full long names)
# =====================================================================
CUSTOM_CELLTYPE_COLORS = {
    "Astrocytes": "#555555",
    "Claustrum Neurons": "#EC8334",
    "Corticothalamic L6": "#DA9028",
    "Endothelial Cells": "#C39B2D",
    "Excitatory L5": "#AAA636",
    "Interneuron L2/3": "#86AE3F",
    "Interneuron L4": "#54B64A",
    "Interneuron L5": "#2DB34B",
    "Interneuron L6": "#2CB66D",
    "Layer 6b Neurons": "#2BBA95",
    "LAMP5 Interneurons": "#18BCB3",
    "Microglia": "#1DBCD1",
    "NDNF Interneurons": "#1EB5E9",
    "Near-Projecting Neurons": "#3CA4DC",
    "Oligodendrocytes": "#6F95CE",
    "Oligodendrocyte Precursor Cells": "#9588C1",
    "Parvalbumin Interneurons": "#AC7EB8",
    "Parvalbumin Chandelier Cells": "#C275B1",
    "Somatostatin-Expressing Neurons": "#D66DAB",
    "VIP Interneurons": "#EE68A7",
    "Vascular Leptomeningeal Cells": "#F06A95",
    "Astrocytes WDR49": "#FF0000",
}

SHORT_LABELS = {
    "Astrocytes": "ASC",
    "Claustrum Neurons": "CLA",
    "Corticothalamic L6": "CT L6",
    "Endothelial Cells": "Endo",
    "Excitatory L5": "ET L5",
    "Interneuron L2/3": "IT L2/3",
    "Interneuron L4": "IT L4",
    "Interneuron L5": "IT L5",
    "Interneuron L6": "IT L6",
    "Layer 6b Neurons": "L6b",
    "LAMP5 Interneurons": "LAMP5",
    "Microglia": "MGC",
    "NDNF Interneurons": "NDNF",
    "Near-Projecting Neurons": "NP",
    "Oligodendrocytes": "ODC",
    "Oligodendrocyte Precursor Cells": "OPC",
    "Parvalbumin Interneurons": "PVALB",
    "Parvalbumin Chandelier Cells": "PVALB ChC",
    "Somatostatin-Expressing Neurons": "SST",
    "VIP Interneurons": "VIP",
    "Vascular Leptomeningeal Cells": "VLMC",
    "Astrocytes WDR49": "ASC WDR49",
}

FALLBACK_CELLTYPE_COLOR = "#B0B0B0"
NAN_CELLTYPE_COLOR = "#D0D0D0"


# =====================================================================
# Contrast definitions
# =====================================================================
@dataclass
class Contrast:
    name: str
    label: str
    keep_statuses: list[str]
    control_level: str
    status_col: str
    formula: str | None = None
    extra_covariates: list[str] = field(default_factory=list)
    covariate_obs: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.covariate_obs:
            self.covariate_obs = [self.status_col] + self.extra_covariates


COLLAPSED_COL = "type_als_collapsed"

CONTRASTS = [
    Contrast(
        name="ALS_vs_Control",
        label="ALS (C9orf72 + Sporadic) vs Control",
        keep_statuses=["ALS", "Control"],
        control_level="Control",
        status_col=COLLAPSED_COL,
    ),
    Contrast(
        name="C9orf72_vs_Control",
        label="C9orf72 vs Control",
        keep_statuses=["C9orf72", "Control"],
        control_level="Control",
        status_col=STATUS_SOURCE_COL,
    ),
    Contrast(
        name="Sporadic_vs_Control",
        label="Sporadic vs Control",
        keep_statuses=["Sporadic", "Control"],
        control_level="Control",
        status_col=STATUS_SOURCE_COL,
    ),
    Contrast(
        name="C9orf72_vs_Sporadic",
        label="C9orf72 vs Sporadic",
        keep_statuses=["C9orf72", "Sporadic"],
        control_level="Sporadic",
        status_col=STATUS_SOURCE_COL,
    ),
    Contrast(
        name="C9orf72_vs_Sporadic_survAdj",
        label="C9orf72 vs Sporadic (Survival-adjusted)",
        keep_statuses=["C9orf72", "Sporadic"],
        control_level="Sporadic",
        status_col=STATUS_SOURCE_COL,
        # RAW covariate, and this is the bug. Not standardised, which collapses the
        # posterior. Superseded by sccoda_refit_scaled_c6_c7.py. Kept unedited so the
        # published run stays reproducible.
        formula="status_ref + Survival",
        extra_covariates=[SURVIVAL_COL],
    ),
    Contrast(
        name="ALS_survival_only",
        label="ALS samples: Survival as sole covariate",
        keep_statuses=["C9orf72", "Sporadic"],
        control_level="Sporadic",
        status_col=STATUS_SOURCE_COL,
        # Same raw-covariate problem as the contrast above. Superseded.
        formula="Survival",
        extra_covariates=[SURVIVAL_COL],
    ),
    Contrast(
        name="All_AgeAtDeath",
        label="All samples: Age at Death as covariate",
        keep_statuses=["C9orf72", "Sporadic", "Control"],
        control_level="Control",
        status_col=STATUS_SOURCE_COL,
        # Same raw-covariate problem, and separately the wrong cohort scope.
        # Superseded twice over: by the refit, then by sccoda_multiome_age_wdr49.py.
        formula="status_ref + Age_At_Death",
        extra_covariates=[AGE_COL],
    ),
]


# =====================================================================
# Plot settings
# =====================================================================
PLOT_DPI = 600
PLOT_FONTSIZE = 8
FIG_W = 6.0
FIG_H = 3.2

BOXPLOT_SHOW_POINTS = True
BOXPLOT_POINT_SIZE = 10
BOXPLOT_ALPHA = 0.7


# =====================================================================
# Utilities
# =====================================================================
def _safe_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def _save_df(df, stem: Path) -> None:
    if df is None:
        return
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return
    df.to_csv(str(stem) + ".csv", index=True)
    df.to_pickle(str(stem) + ".pkl")


def _flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def _set_nature_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": PLOT_FONTSIZE,
        "axes.labelsize": PLOT_FONTSIZE,
        "axes.titlesize": PLOT_FONTSIZE + 1,
        "xtick.labelsize": PLOT_FONTSIZE - 1,
        "ytick.labelsize": PLOT_FONTSIZE - 1,
        "legend.fontsize": PLOT_FONTSIZE - 1,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False))


def _save_figure(fig, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(outbase) + ".pdf", bbox_inches="tight")
    fig.savefig(str(outbase) + ".png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _as_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _get_celltype_names(adata_mod) -> list[str]:
    try:
        return list(adata_mod.var_names.astype(str))
    except Exception:
        return [f"ct_{i}" for i in range(adata_mod.shape[1])]


# =====================================================================
# Colour / label helpers
# =====================================================================
def _celltype_color(celltype: str) -> str:
    if celltype is None or str(celltype).strip().lower() == "nan":
        return NAN_CELLTYPE_COLOR
    ct = str(celltype).strip()
    if ct in CUSTOM_CELLTYPE_COLORS:
        return CUSTOM_CELLTYPE_COLORS[ct]
    lower_map = {k.lower(): v for k, v in CUSTOM_CELLTYPE_COLORS.items()}
    if ct.lower() in lower_map:
        return lower_map[ct.lower()]
    return FALLBACK_CELLTYPE_COLOR


def _short_label(celltype: str) -> str:
    if celltype is None:
        return "nan"
    ct = str(celltype).strip()
    if ct in SHORT_LABELS:
        return SHORT_LABELS[ct]
    lower_map = {k.lower(): v for k, v in SHORT_LABELS.items()}
    if ct.lower() in lower_map:
        return lower_map[ct.lower()]
    return ct


def _celltype_colors_list(celltypes: list[str]) -> list[str]:
    return [_celltype_color(ct) for ct in celltypes]


# =====================================================================
# Status helpers
# =====================================================================
def _get_status_series(adata_mod, status_col: str) -> pd.Series:
    for col in [status_col, "status_ref", STATUS_SOURCE_COL, COLLAPSED_COL]:
        if col in adata_mod.obs.columns:
            return adata_mod.obs[col].astype(str)
    raise ValueError(f"Could not find status column '{status_col}' in modality .obs.")


def _force_control_reference(adata_mod, status_col, control_level, all_levels):
    obs = adata_mod.obs
    ref_col = "status_ref"
    present = sorted(obs[status_col].astype(str).unique().tolist())
    ordered = [control_level] + [x for x in all_levels if x != control_level and x in present]
    extras = [x for x in present if x not in ordered]
    categories = ordered + extras
    obs[ref_col] = pd.Categorical(obs[status_col].astype(str), categories=categories, ordered=True)
    adata_mod.obs = obs
    return ref_col


# =====================================================================
# Composition helpers
# =====================================================================
def _compute_per_sample_composition_df(adata_mod, status_col, control_level, keep_statuses):
    X = _as_dense(adata_mod.X)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    comp = X / row_sums

    status = _get_status_series(adata_mod, status_col).values.astype(str)
    samples = adata_mod.obs_names.astype(str)
    celltypes = _get_celltype_names(adata_mod)

    wide = pd.DataFrame(comp, index=samples, columns=celltypes)
    wide["status"] = status
    wide["sample"] = samples

    long = wide.melt(id_vars=["sample", "status"], var_name="cell_type", value_name="composition")
    preferred = [control_level] + [s for s in keep_statuses if s != control_level]
    long["status"] = pd.Categorical(long["status"].astype(str), categories=preferred, ordered=True)
    return long, wide


def _compute_mean_composition_by_status(adata_mod, status_col, control_level, keep_statuses):
    X = _as_dense(adata_mod.X)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    comp = X / row_sums

    status = _get_status_series(adata_mod, status_col)
    celltypes = _get_celltype_names(adata_mod)

    df = pd.DataFrame(comp, columns=celltypes, index=adata_mod.obs_names.astype(str))
    df["status"] = status.values
    mean_df = df.groupby("status", observed=True).mean(numeric_only=True)

    preferred = [control_level] + [s for s in keep_statuses if s != control_level]
    ordered = [s for s in preferred if s in mean_df.index]
    rest = [s for s in mean_df.index if s not in ordered]
    mean_df = mean_df.loc[ordered + rest]
    return mean_df


# =====================================================================
# Plotting functions
# =====================================================================

def _plot_stacked_bar(mean_comp, outbase, title):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_subplot(111)
    statuses = list(mean_comp.index.astype(str))
    celltypes = list(mean_comp.columns.astype(str))
    colors = _celltype_colors_list(celltypes)
    bottoms = np.zeros(len(statuses), dtype=float)
    x = np.arange(len(statuses))
    for j, ct in enumerate(celltypes):
        vals = mean_comp[ct].values.astype(float)
        ax.bar(x, vals, bottom=bottoms, width=0.8, color=colors[j],
               edgecolor="white", linewidth=0.3, label=_short_label(ct))
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(statuses, rotation=45, ha="right")
    ax.set_ylabel("Mean composition")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              ncol=1, borderaxespad=0.0, handlelength=1.0, handletextpad=0.4)
    _despine(ax)
    _save_figure(fig, outbase)


def _plot_stacked_bar_per_sample(wide_df, outbase, title, status_col="status"):
    celltypes = [c for c in wide_df.columns if c not in ("status", "sample")]
    colors = _celltype_colors_list(celltypes)
    df = wide_df.sort_values([status_col, "sample"]).reset_index(drop=True)
    n = df.shape[0]
    fig_w = max(FIG_W, n * 0.25 + 2)
    fig = plt.figure(figsize=(fig_w, FIG_H + 0.5))
    ax = fig.add_subplot(111)
    x = np.arange(n)
    bottoms = np.zeros(n, dtype=float)
    for j, ct in enumerate(celltypes):
        vals = df[ct].values.astype(float)
        ax.bar(x, vals, bottom=bottoms, width=1.0, color=colors[j],
               edgecolor="none", label=_short_label(ct))
        bottoms += vals
    statuses = df[status_col].values.astype(str)
    prev = statuses[0]
    group_starts = [0]
    for i in range(1, n):
        if statuses[i] != prev:
            ax.axvline(i - 0.5, color="black", linewidth=1.0)
            group_starts.append(i)
            prev = statuses[i]
    group_starts.append(n)
    for gi in range(len(group_starts) - 1):
        mid = (group_starts[gi] + group_starts[gi + 1]) / 2.0
        ax.text(mid, -0.06, statuses[group_starts[gi]],
                ha="center", va="top", fontsize=PLOT_FONTSIZE - 1,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Relative abundance")
    ax.set_title(title)
    ax.set_xticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              ncol=1, handlelength=1.0, handletextpad=0.4)
    _despine(ax)
    _save_figure(fig, outbase)


def _plot_heatmap(mean_comp, outbase, title):
    data = mean_comp.T
    short_idx = [_short_label(ct) for ct in data.index.astype(str)]
    fig = plt.figure(figsize=(FIG_W, max(FIG_H, 0.18 * data.shape[0] + 1.6)))
    ax = fig.add_subplot(111)
    im = ax.imshow(data.values, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(list(data.columns.astype(str)), rotation=45, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(short_idx)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean composition")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    _save_figure(fig, outbase)


def _plot_sample_composition_heatmap(wide_df, outbase, title, status_col="status"):
    celltypes = [c for c in wide_df.columns if c not in ("status", "sample")]
    short_ct = [_short_label(ct) for ct in celltypes]
    mat = wide_df[celltypes].values.astype(float)
    if mat.shape[0] > 2:
        Z = linkage(mat, method="ward", metric="euclidean")
        order = leaves_list(Z)
    else:
        order = np.arange(mat.shape[0])
    mat_ordered = mat[order]
    status_labels = wide_df[status_col].values.astype(str)[order]
    fig_h = max(FIG_H, 0.14 * mat.shape[0] + 1.5)
    fig, (ax_status, ax_heat) = plt.subplots(
        1, 2, figsize=(FIG_W + 1, fig_h),
        gridspec_kw={"width_ratios": [0.04, 1], "wspace": 0.02})
    unique_st = sorted(set(status_labels))
    st_cmap = {s: plt.cm.Set2(i / max(len(unique_st) - 1, 1)) for i, s in enumerate(unique_st)}
    st_colors = np.array([st_cmap[s] for s in status_labels])
    ax_status.imshow(st_colors.reshape(-1, 1, 4), aspect="auto", interpolation="nearest")
    ax_status.set_xticks([]); ax_status.set_yticks([]); ax_status.set_ylabel("Samples")
    im = ax_heat.imshow(mat_ordered, aspect="auto", interpolation="nearest", cmap="YlOrRd")
    ax_heat.set_xticks(np.arange(len(celltypes)))
    ax_heat.set_xticklabels(short_ct, rotation=90, ha="center", fontsize=PLOT_FONTSIZE - 1)
    ax_heat.set_yticks([]); ax_heat.set_title(title)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
    cbar.set_label("Relative abundance")
    handles = [Patch(facecolor=st_cmap[s], label=s) for s in unique_st]
    ax_status.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.5, -0.02),
                     frameon=False, fontsize=PLOT_FONTSIZE - 2)
    for ax in [ax_status, ax_heat]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
    _save_figure(fig, outbase)


def _plot_all_boxplots(long_df, plotdir, control_level, keep_statuses):
    saved = {}
    boxdir = plotdir / "boxplots_by_celltype"
    boxdir.mkdir(parents=True, exist_ok=True)
    celltypes = sorted(long_df["cell_type"].astype(str).unique().tolist())
    statuses = [control_level] + [s for s in keep_statuses if s != control_level]
    for ct in celltypes:
        sub = long_df[long_df["cell_type"].astype(str) == ct].copy()
        if sub.shape[0] == 0:
            continue
        grouped, labels = [], []
        for st in statuses:
            vals = sub.loc[sub["status"].astype(str) == st, "composition"].values.astype(float)
            grouped.append(vals); labels.append(st)
        fig = plt.figure(figsize=(3.6, 2.6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(grouped, labels=labels, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"linewidth": 1.0, "color": "black"},
                        boxprops={"linewidth": 0.8, "edgecolor": "black"},
                        whiskerprops={"linewidth": 0.8, "color": "black"},
                        capprops={"linewidth": 0.8, "color": "black"})
        ct_color = _celltype_color(ct)
        for patch in bp["boxes"]:
            patch.set_facecolor(ct_color); patch.set_alpha(0.55)
        if BOXPLOT_SHOW_POINTS:
            rng = np.random.default_rng(0)
            for i, vals in enumerate(grouped, start=1):
                if vals.size == 0: continue
                jitter = rng.normal(0.0, 0.04, size=vals.size)
                ax.scatter(i + jitter, vals, s=BOXPLOT_POINT_SIZE, alpha=BOXPLOT_ALPHA,
                           linewidths=0, color=ct_color)
        ax.set_title(f"{_short_label(ct)}  ({ct})")
        ax.set_ylabel("Relative abundance (per sample)")
        ax.set_xticklabels(labels, rotation=45, ha="right")
        _despine(ax)
        safe_ct = str(ct).replace(os.sep, "_").replace(" ", "_").replace("/", "_")
        outbase = boxdir / f"boxplot__{safe_ct}"
        _save_figure(fig, outbase)
        saved[f"boxplot__{ct}"] = str(outbase)
    return saved


def _plot_boxplot_panel(long_df, outbase, title, control_level, keep_statuses,
                        credible_celltypes=None):
    celltypes = sorted(long_df["cell_type"].astype(str).unique().tolist())
    statuses = [control_level] + [s for s in keep_statuses if s != control_level]
    n_ct = len(celltypes)
    ncols = min(6, n_ct)
    nrows = int(np.ceil(n_ct / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.0), squeeze=False)
    for idx, ct in enumerate(celltypes):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = long_df[long_df["cell_type"].astype(str) == ct]
        grouped = []
        for st in statuses:
            vals = sub.loc[sub["status"].astype(str) == st, "composition"].values.astype(float)
            grouped.append(vals)
        bp = ax.boxplot(grouped, labels=statuses, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"linewidth": 0.8, "color": "black"},
                        boxprops={"linewidth": 0.5}, whiskerprops={"linewidth": 0.5},
                        capprops={"linewidth": 0.5})
        ct_color = _celltype_color(ct)
        for patch in bp["boxes"]:
            patch.set_facecolor(ct_color); patch.set_alpha(0.5)
        rng = np.random.default_rng(42)
        for i, vals in enumerate(grouped, start=1):
            if vals.size > 0:
                jitter = rng.normal(0, 0.04, size=vals.size)
                ax.scatter(i + jitter, vals, s=6, alpha=0.6, linewidths=0, color=ct_color)
        is_cred = credible_celltypes and ct in credible_celltypes
        ax.set_title(_short_label(ct), fontsize=PLOT_FONTSIZE - 1,
                     fontweight="bold" if is_cred else "normal",
                     color="red" if is_cred else "black")
        ax.tick_params(labelsize=PLOT_FONTSIZE - 2)
        if r < nrows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(statuses, rotation=45, ha="right", fontsize=PLOT_FONTSIZE - 2)
        _despine(ax)
    for idx in range(n_ct, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(title, fontsize=PLOT_FONTSIZE + 1, y=1.01)
    _save_figure(fig, outbase)


def _plot_proportion_scatter(wide_df, outbase, title, status_col="status"):
    statuses = sorted(wide_df[status_col].unique())
    if len(statuses) != 2:
        return
    celltypes = [c for c in wide_df.columns if c not in ("status", "sample")]
    g1 = wide_df[wide_df[status_col] == statuses[0]][celltypes].mean()
    g2 = wide_df[wide_df[status_col] == statuses[1]][celltypes].mean()
    fig = plt.figure(figsize=(FIG_W, FIG_W))
    ax = fig.add_subplot(111)
    colors = _celltype_colors_list(celltypes)
    ax.scatter(g1.values, g2.values, c=colors, s=50, edgecolors="black", linewidths=0.4, zorder=3)
    lim = max(g1.max(), g2.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.6, alpha=0.4)
    for i, ct in enumerate(celltypes):
        ax.annotate(_short_label(ct), (g1.values[i], g2.values[i]),
                    fontsize=PLOT_FONTSIZE - 2, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"Mean proportion: {statuses[0]}")
    ax.set_ylabel(f"Mean proportion: {statuses[1]}")
    ax.set_title(title)
    _despine(ax)
    _save_figure(fig, outbase)


# =====================================================================
# Effect column inference
# =====================================================================
def _infer_effect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols: return cols[c]
        return None
    return {
        "effect": pick("effect", "mean", "coef", "estimate", "log2_fold_change",
                        "log_fold_change", "beta"),
        "lower": pick("hdi_2.5%", "hdi_2.5", "lower", "ci_lower",
                       "credible_interval_lower", "q2.5", "2.5%"),
        "upper": pick("hdi_97.5%", "hdi_97.5", "upper", "ci_upper",
                       "credible_interval_upper", "q97.5", "97.5%"),
        "prob": pick("inclusion_prob", "inclusion_probability", "posterior_prob", "prob"),
        "credible": pick("credible", "is_credible", "credible_effect", "significant", "selected"),
        "cell_type": pick("cell_type", "celltype", "feature", "taxon", "cell"),
        "covariate": pick("covariate", "condition", "group", "level", "factor"),
    }


def _ensure_celltype_column(df, colname):
    out = df.copy()
    if colname and colname in out.columns: return out
    if out.index.name and "cell" in str(out.index.name).lower():
        out["cell_type"] = out.index.astype(str); return out
    if isinstance(out.index, pd.MultiIndex):
        out["cell_type"] = out.index.get_level_values(-1).astype(str)
        out["covariate"] = out.index.get_level_values(0).astype(str)
        return out
    out["cell_type"] = out.index.astype(str)
    return out


def _ensure_covariate_column(df, colname):
    out = df.copy()
    if colname and colname in out.columns: return out
    if isinstance(out.index, pd.MultiIndex):
        out["covariate"] = out.index.get_level_values(0).astype(str); return out
    out["covariate"] = "all"
    return out


def _plot_effects_by_covariate(df, outdir, stem, title_prefix):
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        return
    inferred = _infer_effect_columns(df)
    effect_col = inferred["effect"]
    if effect_col is None: return
    df2 = df.copy()
    df2 = _ensure_celltype_column(df2, inferred["cell_type"])
    df2 = _ensure_covariate_column(df2, inferred["covariate"])
    lower_col, upper_col = inferred["lower"], inferred["upper"]
    prob_col, cred_col = inferred["prob"], inferred["credible"]
    df2[effect_col] = pd.to_numeric(df2[effect_col], errors="coerce")
    if lower_col and lower_col in df2.columns:
        df2[lower_col] = pd.to_numeric(df2[lower_col], errors="coerce")
    if upper_col and upper_col in df2.columns:
        df2[upper_col] = pd.to_numeric(df2[upper_col], errors="coerce")
    if prob_col and prob_col in df2.columns:
        df2[prob_col] = pd.to_numeric(df2[prob_col], errors="coerce")
    if cred_col and cred_col in df2.columns:
        if df2[cred_col].dtype != bool:
            df2[cred_col] = df2[cred_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    else:
        cred_col = None
    for cov, sub in df2.groupby("covariate", observed=True):
        sub = sub.dropna(subset=[effect_col]).copy()
        if sub.shape[0] == 0: continue
        sub["abs_effect"] = sub[effect_col].abs()
        sub = sub.sort_values("abs_effect", ascending=False)
        if sub.shape[0] > 60: sub = sub.iloc[:60].copy()
        sub = sub.iloc[::-1]
        ylabels_long = sub["cell_type"].astype(str).tolist()
        ylabels = [_short_label(ct) for ct in ylabels_long]
        y = np.arange(len(ylabels))
        effects = sub[effect_col].values.astype(float)
        colors = _celltype_colors_list(ylabels_long)
        fig_h = max(FIG_H, 0.16 * len(ylabels) + 1.3)
        fig = plt.figure(figsize=(FIG_W, fig_h))
        ax = fig.add_subplot(111)
        ax.barh(y, effects, height=0.75, edgecolor="black", linewidth=0.3, color=colors)
        if lower_col and upper_col and (lower_col in sub.columns) and (upper_col in sub.columns):
            lower = sub[lower_col].values.astype(float)
            upper = sub[upper_col].values.astype(float)
            xerr = np.vstack([effects - lower, upper - effects])
            ax.errorbar(effects, y, xerr=xerr, fmt="none", ecolor="black",
                        elinewidth=0.6, capsize=2, capthick=0.6)
        ax.axvline(0.0, linewidth=0.8, color="black")
        ax.set_yticks(y); ax.set_yticklabels(ylabels)
        ax.set_xlabel(effect_col)
        ax.set_title(f"{title_prefix}: {cov}")
        if prob_col and prob_col in sub.columns:
            probs = sub[prob_col].values
            for yi, (xe, pr) in enumerate(zip(effects, probs)):
                if np.isfinite(pr):
                    ax.text(xe + (0.01 if xe >= 0 else -0.01), yi, f"{pr:.2f}",
                            va="center", ha="left" if xe >= 0 else "right",
                            fontsize=PLOT_FONTSIZE - 2)
        if cred_col:
            credible_mask = sub[cred_col].values.astype(bool)
            if credible_mask.any():
                ax.scatter(np.where(credible_mask, effects, np.nan), y, s=12,
                           marker="s", edgecolors="black", linewidths=0.4,
                           facecolors="none", zorder=3)
                ax.text(0.99, 0.01, "Squares = credible", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=PLOT_FONTSIZE - 2)
        _despine(ax)
        safe_cov = str(cov).replace(os.sep, "_").replace(" ", "_")
        outbase = outdir / f"{stem}__{safe_cov}"
        _save_figure(fig, outbase)


def _plot_lfc_dotplot(summary_df, outbase, title):
    if summary_df is None or summary_df.shape[0] == 0: return
    inferred = _infer_effect_columns(summary_df)
    effect_col = inferred["effect"]
    prob_col, cred_col = inferred["prob"], inferred["credible"]
    if effect_col is None: return
    df = summary_df.copy()
    df = _ensure_celltype_column(df, inferred["cell_type"])
    df[effect_col] = pd.to_numeric(df[effect_col], errors="coerce")
    df = df.dropna(subset=[effect_col])
    if prob_col and prob_col in df.columns:
        df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    else:
        prob_col = None
    celltypes_long = df["cell_type"].astype(str).values
    celltypes_short = [_short_label(ct) for ct in celltypes_long]
    effects = df[effect_col].values.astype(float)
    fig = plt.figure(figsize=(FIG_W, max(FIG_H, 0.2 * len(celltypes_long) + 1.0)))
    ax = fig.add_subplot(111)
    y = np.arange(len(celltypes_short))
    if prob_col and prob_col in df.columns:
        probs = df[prob_col].values.astype(float)
        scatter = ax.scatter(effects, y, c=probs, cmap="RdYlGn", vmin=0, vmax=1,
                             s=40, edgecolors="black", linewidths=0.4, zorder=3)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Inclusion probability")
    else:
        colors = _celltype_colors_list(celltypes_long)
        ax.scatter(effects, y, c=colors, s=40, edgecolors="black", linewidths=0.4, zorder=3)
    if cred_col and cred_col in df.columns:
        cred_mask = df[cred_col].astype(str).str.lower().isin(["true", "1", "yes"])
        if cred_mask.any():
            ax.scatter(effects[cred_mask], y[cred_mask.values], s=100, marker="*",
                       color="red", zorder=4, label="Credible")
            ax.legend(loc="lower right", fontsize=PLOT_FONTSIZE - 1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(celltypes_short)
    ax.set_xlabel(f"Effect size ({effect_col})")
    ax.set_title(title)
    _despine(ax)
    _save_figure(fig, outbase)


def _try_pertpy_builtin_plots(sccoda_model, sccoda_data, modality_key, plotdir, contrast_label):
    # Try without ax= first (newer pertpy), then with ax= (older pertpy)
    for plot_name, plot_func_name, needs_feature in [
        ("pertpy_stacked_barplot", "plot_stacked_barplot", True),
        ("pertpy_boxplots", "plot_boxplots", True),
        ("pertpy_effects_barplot", "plot_effects_barplot", False),
    ]:
        func = getattr(sccoda_model, plot_func_name, None)
        if func is None:
            _flush_print(f"    [SKIP] {plot_name}: method not found")
            continue
        try:
            kwargs = {"modality_key": modality_key}
            if needs_feature:
                kwargs["feature_name"] = "status_ref"
            result = func(sccoda_data, **kwargs)
            if result is not None:
                if isinstance(result, matplotlib.figure.Figure):
                    _save_figure(result, plotdir / plot_name)
                else:
                    fig = plt.gcf()
                    _save_figure(fig, plotdir / plot_name)
            _flush_print(f"    [OK] {plot_name}")
        except Exception as e:
            _flush_print(f"    [SKIP] {plot_name}: {e}")


# =====================================================================
# Run one contrast
# =====================================================================
def run_contrast(adata_full, contrast, root_outdir, manifest):
    cname = contrast.name
    _flush_print(f"\n{'='*70}")
    _flush_print(f"[CONTRAST] {cname}: {contrast.label}")
    _flush_print(f"  statuses = {contrast.keep_statuses}, control = {contrast.control_level}")
    _flush_print(f"  status_col = {contrast.status_col}")
    _flush_print(f"  formula = {contrast.formula}")
    _flush_print(f"  extra_covariates = {contrast.extra_covariates}")
    _flush_print(f"  FDR level = {FDR_LEVEL}")
    _flush_print(f"{'='*70}")

    c_outdir = root_outdir / cname
    c_outdir.mkdir(parents=True, exist_ok=True)
    c_plotdir = c_outdir / "plots"
    c_plotdir.mkdir(parents=True, exist_ok=True)

    c_manifest = {
        "contrast_name": cname,
        "contrast_label": contrast.label,
        "keep_statuses": contrast.keep_statuses,
        "control_level": contrast.control_level,
        "status_col": contrast.status_col,
        "formula": contrast.formula,
        "extra_covariates": contrast.extra_covariates,
        "fdr_level": FDR_LEVEL,
        "saved": {},
        "notes": [],
    }

    # --- Subset cells ---
    status_series = adata_full.obs[contrast.status_col].astype(str)
    keep_mask = status_series.isin(contrast.keep_statuses).values
    if int(keep_mask.sum()) == 0:
        msg = f"[SKIP] 0 cells for contrast {cname}."
        _flush_print(msg)
        c_manifest["notes"].append(msg)
        manifest["contrasts"][cname] = c_manifest
        return

    adata_sub = adata_full[keep_mask].copy()

    contrast_status_key = f"_cstatus_{cname}"
    adata_sub.obs[contrast_status_key] = adata_sub.obs[contrast.status_col].astype(str)

    for ec in contrast.extra_covariates:
        if ec not in adata_sub.obs.columns:
            msg = f"[SKIP] Extra covariate '{ec}' missing. Skipping {cname}."
            _flush_print(msg)
            c_manifest["notes"].append(msg)
            manifest["contrasts"][cname] = c_manifest
            return
        nan_mask = adata_sub.obs[ec].isna()
        if nan_mask.any():
            n_drop = int(nan_mask.sum())
            _flush_print(f"  Dropping {n_drop} cells with NaN in '{ec}'")
            adata_sub = adata_sub[~nan_mask.values].copy()
        adata_sub.obs[ec] = pd.to_numeric(adata_sub.obs[ec], errors="coerce")

    _flush_print(f"  Cells retained: {adata_sub.n_obs}")

    # --- Build scCODA MuData ---
    _flush_print(f"  Building scCODA MuData …")
    sccoda_model = pt.tl.Sccoda()

    covariate_obs_list = [contrast_status_key] + contrast.extra_covariates
    seen = set()
    covariate_obs_list = [x for x in covariate_obs_list if not (x in seen or seen.add(x))]

    sccoda_data = sccoda_model.load(
        adata_sub, type="cell_level", generate_sample_level=True,
        cell_type_identifier=CELL_TYPE_COL, sample_identifier=SAMPLE_COL,
        covariate_obs=covariate_obs_list)

    modality_key = f"coda_{cname}"
    coda = sccoda_data["coda"]

    coda_status_col = contrast_status_key
    if coda_status_col not in coda.obs.columns:
        for col in coda.obs.columns:
            if contrast_status_key in col or cname in col:
                coda_status_col = col
                break

    mask = coda.obs[coda_status_col].astype(str).isin(contrast.keep_statuses).values
    if int(mask.sum()) == 0:
        msg = f"[SKIP] 0 samples after subset for {cname}."
        _flush_print(msg)
        c_manifest["notes"].append(msg)
        manifest["contrasts"][cname] = c_manifest
        return

    sccoda_data.mod[modality_key] = coda[mask].copy()

    n_samples = sccoda_data.mod[modality_key].n_obs
    n_celltypes = sccoda_data.mod[modality_key].n_vars
    _flush_print(f"  Samples in coda: {n_samples}, cell types: {n_celltypes}")
    c_manifest["n_samples"] = n_samples
    c_manifest["n_celltypes"] = n_celltypes

    ref_col = _force_control_reference(
        sccoda_data.mod[modality_key], coda_status_col,
        contrast.control_level, contrast.keep_statuses)

    formula_str = contrast.formula if contrast.formula else ref_col
    _flush_print(f"  Formula: {formula_str}")
    c_manifest["formula_used"] = formula_str

    # --- Prepare + Run ---
    _flush_print(f"  Preparing design matrix …")
    sccoda_data = sccoda_model.prepare(
        sccoda_data, modality_key=modality_key, formula=formula_str)

    _flush_print(f"  Running NUTS (CPU) …")
    # No sampler arguments, so pertpy's defaults apply: 10,000 draws after 1,000
    # warmup, seed 0. State this in Methods, since the call does not show it.
    sccoda_model.run_nuts(sccoda_data, modality_key=modality_key)

    # --- Save model ---
    try:
        model_pkl = c_outdir / "sccoda_model.pkl"
        with model_pkl.open("wb") as f:
            pickle.dump(sccoda_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        c_manifest["saved"]["model_pkl"] = str(model_pkl)
        _flush_print(f"  Saved: {model_pkl}")
    except Exception as e:
        c_manifest["notes"].append(f"Failed to pickle model: {repr(e)}")

    # --- Summary + Credible effects at FDR_LEVEL ---
    _flush_print(f"  Computing summary + credible effects (FDR={FDR_LEVEL}) …")
    summary_res = None
    cred_res = None

    try:
        summary_res = sccoda_model.summary(sccoda_data, modality_key=modality_key)
        _save_df(summary_res, c_outdir / "summary")
        c_manifest["saved"]["summary_csv"] = str(c_outdir / "summary.csv")
        _flush_print(f"  Saved: summary.csv/.pkl")
    except Exception as e:
        c_manifest["notes"].append(f"Failed summary: {repr(e)}")
        _flush_print(f"  [WARN] summary failed: {e}")

    try:
        cred_res = sccoda_model.credible_effects(
            sccoda_data, modality_key=modality_key, est_fdr=FDR_LEVEL)
        _save_df(cred_res, c_outdir / "credible_effects")
        c_manifest["saved"]["credible_effects_csv"] = str(c_outdir / "credible_effects.csv")
        _flush_print(f"  Saved: credible_effects.csv/.pkl  (FDR={FDR_LEVEL})")
    except Exception as e:
        c_manifest["notes"].append(f"Failed credible_effects: {repr(e)}")
        _flush_print(f"  [WARN] credible_effects failed: {e}")

    # --- Determine credible cell types ---
    credible_celltypes = set()
    if isinstance(cred_res, pd.DataFrame) and cred_res.shape[0] > 0:
        inferred = _infer_effect_columns(cred_res)
        cred_col_name = inferred["credible"]
        ct_col_name = inferred["cell_type"]
        if cred_col_name and ct_col_name:
            tmp = _ensure_celltype_column(cred_res, ct_col_name)
            mask_cred = tmp[cred_col_name].astype(str).str.lower().isin(["true", "1", "yes"])
            credible_celltypes = set(tmp.loc[mask_cred, "cell_type"].astype(str).tolist())
        else:
            tmp = _ensure_celltype_column(cred_res, None)
            credible_celltypes = set(tmp["cell_type"].astype(str).tolist())

    _flush_print(f"  Credible cell types (FDR={FDR_LEVEL}): "
                 f"{credible_celltypes if credible_celltypes else 'none detected'}")

    # --- Save MuData ---
    try:
        post_h5mu = c_outdir / f"sccoda_data_{cname}_postrun.h5mu"
        sccoda_data.write(post_h5mu)
        c_manifest["saved"]["postrun_h5mu"] = str(post_h5mu)
        _flush_print(f"  Saved: {post_h5mu}")
    except Exception as e:
        c_manifest["notes"].append(f"Failed to write postrun h5mu: {repr(e)}")

    # =================================================================
    # PLOTS
    # =================================================================
    _flush_print(f"  Generating plots …")
    subset = sccoda_data[modality_key]

    try:
        mean_comp = _compute_mean_composition_by_status(
            subset, coda_status_col, contrast.control_level, contrast.keep_statuses)
        mean_comp.to_csv(c_outdir / "mean_composition_by_status.csv", index=True)
        long_df, wide_df = _compute_per_sample_composition_df(
            subset, coda_status_col, contrast.control_level, contrast.keep_statuses)
        long_df.to_csv(c_outdir / "per_sample_composition_long.csv", index=False)
        wide_df.to_csv(c_outdir / "per_sample_composition_wide.csv", index=True)
    except Exception as e:
        c_manifest["notes"].append(f"Failed computing compositions: {repr(e)}")
        _flush_print(f"  [WARN] composition computation failed: {e}")
        manifest["contrasts"][cname] = c_manifest
        return

    try:
        _plot_stacked_bar(mean_comp, c_plotdir / "mean_composition_stacked",
                          title=f"Mean composition: {contrast.label}")
        _flush_print("    [OK] 1. mean stacked bar")
    except Exception as e:
        c_manifest["notes"].append(f"Failed stacked bar: {repr(e)}")

    try:
        _plot_stacked_bar_per_sample(wide_df, c_plotdir / "sample_stacked_bar",
                                     title=f"Per-sample composition: {contrast.label}")
        _flush_print("    [OK] 2. per-sample stacked bar")
    except Exception as e:
        c_manifest["notes"].append(f"Failed sample stacked bar: {repr(e)}")

    try:
        _plot_heatmap(mean_comp, c_plotdir / "mean_composition_heatmap",
                      title=f"Mean composition heatmap: {contrast.label}")
        _flush_print("    [OK] 3. mean heatmap")
    except Exception as e:
        c_manifest["notes"].append(f"Failed heatmap: {repr(e)}")

    try:
        _plot_sample_composition_heatmap(wide_df, c_plotdir / "sample_composition_heatmap",
                                         title=f"Sample composition (clustered): {contrast.label}")
        _flush_print("    [OK] 4. sample clustered heatmap")
    except Exception as e:
        c_manifest["notes"].append(f"Failed sample heatmap: {repr(e)}")

    try:
        box_saved = _plot_all_boxplots(long_df, c_plotdir, contrast.control_level, contrast.keep_statuses)
        c_manifest["saved"]["n_boxplots"] = len(box_saved)
        _flush_print(f"    [OK] 5. {len(box_saved)} individual boxplots")
    except Exception as e:
        c_manifest["notes"].append(f"Failed individual boxplots: {repr(e)}")

    try:
        _plot_boxplot_panel(long_df, c_plotdir / "boxplot_panel",
                            title=f"Cell type compositions (FDR={FDR_LEVEL}): {contrast.label}",
                            control_level=contrast.control_level,
                            keep_statuses=contrast.keep_statuses,
                            credible_celltypes=credible_celltypes)
        _flush_print("    [OK] 6. combined boxplot panel")
    except Exception as e:
        c_manifest["notes"].append(f"Failed boxplot panel: {repr(e)}")

    try:
        if len(contrast.keep_statuses) == 2:
            _plot_proportion_scatter(wide_df, c_plotdir / "proportion_scatter",
                                      title=f"Mean proportion scatter: {contrast.label}")
            _flush_print("    [OK] 7. proportion scatter")
    except Exception as e:
        c_manifest["notes"].append(f"Failed proportion scatter: {repr(e)}")

    try:
        if isinstance(summary_res, pd.DataFrame) and summary_res.shape[0] > 0:
            _plot_effects_by_covariate(summary_res, c_plotdir,
                                       stem="summary_effects",
                                       title_prefix=f"scCODA summary FDR={FDR_LEVEL} ({contrast.label})")
            _flush_print("    [OK] 8. summary forest plot")
    except Exception as e:
        c_manifest["notes"].append(f"Failed summary effects plot: {repr(e)}")

    try:
        if isinstance(cred_res, pd.DataFrame) and cred_res.shape[0] > 0:
            _plot_effects_by_covariate(cred_res, c_plotdir,
                                       stem="credible_effects",
                                       title_prefix=f"scCODA credible FDR={FDR_LEVEL} ({contrast.label})")
            _flush_print("    [OK] 9. credible effects forest plot")
    except Exception as e:
        c_manifest["notes"].append(f"Failed credible effects plot: {repr(e)}")

    try:
        if isinstance(summary_res, pd.DataFrame) and summary_res.shape[0] > 0:
            _plot_lfc_dotplot(summary_res, c_plotdir / "lfc_dotplot",
                              title=f"Effect sizes (FDR={FDR_LEVEL}): {contrast.label}")
            _flush_print("    [OK] 10. LFC dot plot")
    except Exception as e:
        c_manifest["notes"].append(f"Failed LFC dotplot: {repr(e)}")

    try:
        _try_pertpy_builtin_plots(sccoda_model, sccoda_data, modality_key,
                                  c_plotdir, contrast.label)
    except Exception as e:
        c_manifest["notes"].append(f"Failed pertpy built-in plots: {repr(e)}")

    manifest["contrasts"][cname] = c_manifest
    _flush_print(f"  [DONE] Contrast {cname}")


# =====================================================================
# Main
# =====================================================================
def main():
    _set_nature_style()

    t0 = time.time()
    outdir = Path(OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_h5ad": IN_H5AD,
        "outdir": str(outdir),
        "cell_type_col": CELL_TYPE_COL,
        "sample_col": SAMPLE_COL,
        "status_source_col": STATUS_SOURCE_COL,
        "fdr_level": FDR_LEVEL,
        "contrasts_requested": [c.name for c in CONTRASTS],
        "start_time_unix": t0,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "contrasts": {},
        "notes": [],
    }

    _flush_print(f"[INFO] FDR threshold: {FDR_LEVEL}")
    _flush_print(f"[INFO] Output directory: {OUTDIR}")
    _flush_print(f"[INFO] Reading AnnData: {IN_H5AD}")
    adata = sc.read(IN_H5AD)

    for col in [CELL_TYPE_COL, SAMPLE_COL, STATUS_SOURCE_COL]:
        if col not in adata.obs.columns:
            raise ValueError(f"Required obs column missing: {col}")

    adata.obs[STATUS_SOURCE_COL] = adata.obs[STATUS_SOURCE_COL].astype(str)

    collapse_map = {"C9orf72": "ALS", "Sporadic": "ALS", "Control": "Control"}
    adata.obs[COLLAPSED_COL] = adata.obs[STATUS_SOURCE_COL].map(collapse_map)

    for cov_col in [SURVIVAL_COL, AGE_COL]:
        if cov_col in adata.obs.columns:
            adata.obs[cov_col] = pd.to_numeric(adata.obs[cov_col], errors="coerce")
            n_nan = int(adata.obs[cov_col].isna().sum())
            _flush_print(f"[INFO] {cov_col}: {n_nan} NaN values out of {adata.n_obs} cells")

    _flush_print(f"[INFO] Total cells: {adata.n_obs}")
    _flush_print(f"[INFO] Cell type column: {CELL_TYPE_COL}")
    _flush_print(f"[INFO] Unique cell types: {adata.obs[CELL_TYPE_COL].nunique()}")
    _flush_print(f"[INFO] Cell types: {sorted(adata.obs[CELL_TYPE_COL].unique().tolist())}")
    _flush_print(f"[INFO] type_als counts (original):")
    for val, cnt in adata.obs[STATUS_SOURCE_COL].value_counts().items():
        _flush_print(f"        {val}: {cnt}")
    _flush_print(f"[INFO] type_als counts (collapsed):")
    for val, cnt in adata.obs[COLLAPSED_COL].value_counts(dropna=False).items():
        _flush_print(f"        {val}: {cnt}")

    if SURVIVAL_COL in adata.obs.columns:
        _flush_print(f"[INFO] Survival summary (all cells):")
        _flush_print(f"{adata.obs[SURVIVAL_COL].describe()}")
        for grp, sub in adata.obs.groupby(STATUS_SOURCE_COL):
            valid = sub[SURVIVAL_COL].dropna()
            _flush_print(f"  {grp}: n_cells={len(sub)}, survival mean={valid.mean():.1f}, "
                         f"median={valid.median():.1f}, unique_vals={valid.nunique()}")

    if AGE_COL in adata.obs.columns:
        _flush_print(f"[INFO] Age_At_Death summary:")
        _flush_print(f"{adata.obs[AGE_COL].describe()}")

    # ---- Strip heavy objects ----
    _flush_print(f"[INFO] Stripping obsp, obsm, uns to avoid memory explosion on subset …")
    if hasattr(adata, "obsp") and len(adata.obsp) > 0:
        _flush_print(f"  Removing obsp keys: {list(adata.obsp.keys())}")
        for k in list(adata.obsp.keys()):
            del adata.obsp[k]
    if hasattr(adata, "obsm") and len(adata.obsm) > 0:
        _flush_print(f"  Removing obsm keys: {list(adata.obsm.keys())}")
        for k in list(adata.obsm.keys()):
            del adata.obsm[k]
    if hasattr(adata, "uns") and len(adata.uns) > 0:
        _flush_print(f"  Removing uns keys: {list(adata.uns.keys())}")
        adata.uns.clear()
    if hasattr(adata, "varm") and len(adata.varm) > 0:
        for k in list(adata.varm.keys()):
            del adata.varm[k]
    if hasattr(adata, "varp") and len(adata.varp) > 0:
        for k in list(adata.varp.keys()):
            del adata.varp[k]

    import gc
    gc.collect()
    _flush_print(f"[INFO] Stripped. AnnData now: {adata}")

    for cov_col in [SURVIVAL_COL, AGE_COL]:
        if cov_col in adata.obs.columns:
            nuniq = adata.obs.groupby(SAMPLE_COL)[cov_col].nunique()
            bad = nuniq[nuniq > 1]
            if len(bad) > 0:
                _flush_print(f"[WARN] {len(bad)} samples have >1 unique {cov_col} value! "
                             f"First few: {bad.head().to_dict()}")
            else:
                _flush_print(f"[INFO] All samples have a single {cov_col} value — OK")

    # --- Run each contrast ---
    for contrast in CONTRASTS:
        try:
            run_contrast(adata, contrast, outdir, manifest)
        except Exception as e:
            msg = f"[ERROR] Contrast {contrast.name} failed: {repr(e)}"
            _flush_print(msg)
            manifest["notes"].append(msg)
            import traceback
            traceback.print_exc()

    # --- Final manifest ---
    t1 = time.time()
    manifest["end_time_unix"] = t1
    manifest["elapsed_seconds"] = float(t1 - t0)

    manifest_path = outdir / "manifest.json"
    _safe_write_json(manifest, manifest_path)
    _flush_print(f"\n[INFO] Saved: {manifest_path}")
    _flush_print(f"[INFO] Done. Total elapsed: {manifest['elapsed_seconds']:.1f} seconds")
    _flush_print(f"[INFO] Output root: {outdir}")


if __name__ == "__main__":
    main()
