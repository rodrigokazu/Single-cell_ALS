#!/usr/bin/env python3
"""Plot the REAL split-cohort Milo re-analysis results (each cohort's own neighbourhood
graph + DA test, built on Jingtian's real per-cohort embedding) -- SIGNIFICANT
neighbourhoods only (SpatialFDR < ALPHA), not every nhood regardless of significance.
Two panels per contrast: 'Our dataset' (ALS/JCK) and 'Pineda et al' (Pineda2024/Kellis),
each on its own native tSNE (never overlaid -- see the earlier overlay bug/fix).
"""
# ==============================================================================
# WHAT THIS IS
#   The figure generator for the split-cohort re-analysis (milo_split_cohort_
#   run.py's output). Reads whichever cohort h5mu files already exist and draws
#   one two-panel figure per contrast, restricted to neighbourhoods that
#   actually clear SpatialFDR < ALPHA.
#
# WHY SIGNIFICANT-ONLY, UNLIKE THE Merged_v3 COSMETIC PLOTS
#   The earlier full-merged-run plotting script (milo_tsne_logfc_plot_jingtian_
#   v2.py) colours every neighbourhood by logFC regardless of significance,
#   which is defensible there only because it is explicitly labelled internal-
#   viewing-only and every neighbourhood in that run is non-significant anyway.
#   This script is showing a REAL result with REAL significant hits, so the
#   opposite convention applies: drawing every neighbourhood indiscriminately
#   here would bury 655 genuine hits (Pineda2024, Sporadic vs Control) inside a
#   cloud of noise from the other ~14,700 untested-by-eye neighbourhoods.
#   Restricting to SpatialFDR < ALPHA is what makes the figure represent the
#   actual finding rather than an effect-size heatmap with no statistical
#   filter.
#
# WHY IT TOLERATES A MISSING COHORT
#   Written and first run while the ALS/JCK job was still executing on the
#   cluster and Pineda2024's was already done. load_cohort() returns None for
#   any h5mu that doesn't exist yet, and the per-panel loop below draws a plain
#   "not yet available" placeholder for that side rather than failing outright.
#   Re-running this exact script, unmodified, after the second cohort's job
#   finishes is what fills in the other panel -- there is no cohort-specific
#   branch to edit.
#
# WHY EACH PANEL GETS ITS OWN NATIVE tSNE
#   Same reasoning as the fix in milo_tsne_logfc_plot_jingtian_v2.py: Jingtian's
#   ALS and Pineda2024 embeddings are two independently computed spaces with no
#   shared coordinate frame, so each cohort is plotted against its own
#   background and its own neighbourhood positions, never a shared canvas.
# ==============================================================================
# Comments added 28 July 2026. Executable logic is byte-identical to the
# version that ran; a pristine copy sits in _verbatim/, and every non-docstring
# statement was confirmed unchanged with verify_comments_only.py before this
# copy shipped.
# ==============================================================================
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ALPHA = 0.10
CMAP = 'RdBu_r'
TSNE_KEY = 'pc40seuratcc30_tsne'

CONTRASTS = [
    ('ALS_vs_Control',     'ALS vs Control'),
    ('C9orf72_vs_Control', 'C9orf72 vs Control'),
    ('Sporadic_vs_Control','Sporadic vs Control'),
    ('C9orf72_vs_Sporadic','C9orf72 vs Sporadic'),
]
# Each tuple's third element is the h5mu path for that cohort's split-cohort
# run (milo_split_cohort_run.py's own output); load_cohort() below checks for
# its existence rather than assuming it's there.
COHORTS = [
    ('ALS', 'Our dataset',
     '/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Milo/MiloP_results/JingtianSplit_ALS/Milo_ALS.h5mu'),
    ('Pineda2024', 'Pineda et al',
     '/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Milo/MiloP_results/JingtianSplit_Pineda2024/Milo_Pineda2024.h5mu'),
]
OUT_DIR = Path('/home/rodrigok/tmp_check/split_sig_only_plots')
OUT_DIR.mkdir(exist_ok=True, parents=True)


def load_cohort(h5mu_path):
    # Missing file -> None, not an exception. This is the mechanism that lets
    # the script run sensibly while only some cohorts' jobs have finished.
    if not Path(h5mu_path).exists():
        return None
    f = h5py.File(h5mu_path, 'r')
    tsne = f['mod/rna/obsm/' + TSNE_KEY][:]
    cell_idx = f['mod/rna/obs/_index'].asstr()[:]
    index_cell = f['mod/milo/var/index_cell'].asstr()[:]
    annot_grp = f['mod/milo/var/nhood_annotation']
    if hasattr(annot_grp, 'keys'):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in annot_grp['categories'][:]])
        codes = annot_grp['codes'][:]
        annotation = cats[codes]
    else:
        annotation = np.array([x.decode() if isinstance(x, bytes) else x for x in annot_grp[:]])

    contrast_data = {}
    for ck, _ in CONTRASTS:
        lfc_key = f'mod/milo/var/logFC__{ck}'
        fdr_key = f'mod/milo/var/SpatialFDR__{ck}'
        # Guards against a partially-finished job: if a contrast hasn't been
        # fitted yet (or failed), its columns simply aren't in the h5mu, and
        # this cohort is treated as missing that contrast rather than crashing.
        if lfc_key in f and fdr_key in f:
            contrast_data[ck] = {'logFC': f[lfc_key][:], 'SpatialFDR': f[fdr_key][:]}
    f.close()

    # Barcode-keyed join from neighbourhood index-cells to a tSNE row, same
    # pattern used throughout this investigation -- never a positional
    # assumption between the milo and rna modalities.
    cell_pos = {b: i for i, b in enumerate(cell_idx)}
    nhood_rows = np.array([cell_pos.get(c, -1) for c in index_cell])
    valid = nhood_rows >= 0
    nhood_xy = tsne[nhood_rows[valid]]
    nhood_annot = annotation[valid]

    return {
        'bg_xy': tsne,
        'nhood_xy': nhood_xy,
        'nhood_annot': nhood_annot,
        'contrast_data': {ck: {k: v[valid] for k, v in d.items()} for ck, d in contrast_data.items()},
    }


print('Loading cohort data ...')
cohort_data = {}
for key, label, path in COHORTS:
    d = load_cohort(path)
    cohort_data[key] = d
    print(f'  {key}: {"loaded, " + str(d["bg_xy"].shape[0]) + " cells" if d else "NOT AVAILABLE (h5mu missing)"}')

for ck, clabel in CONTRASTS:
    available = [(key, label) for key, label, _ in COHORTS if cohort_data[key] is not None
                 and ck in cohort_data[key]['contrast_data']]
    if not available:
        print(f'[skip] {ck}: no cohort has this contrast available yet')
        continue

    # shared color scale across both panels, from the union of SIGNIFICANT nhoods
    sig_logfc_all = []
    for key, _ in available:
        d = cohort_data[key]['contrast_data'][ck]
        sig_mask = d['SpatialFDR'] < ALPHA
        if sig_mask.any():
            sig_logfc_all.append(d['logFC'][sig_mask])
    if sig_logfc_all:
        allvals = np.concatenate(sig_logfc_all)
        vmax = float(np.nanpercentile(np.abs(allvals), 99))
        if vmax == 0 or not np.isfinite(vmax):
            vmax = float(np.nanmax(np.abs(allvals))) or 1.0
    else:
        # Both cohorts null for this contrast (true for C9orf72_vs_Control and
        # C9orf72_vs_Sporadic in the Pineda2024 result): fall back to a plain
        # +/-1 scale since there's no significant logFC to size it against.
        vmax = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.8), facecolor='white')
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.03, right=0.90, wspace=0.06)
    sc_for_cbar = None

    for ax, (key, label, _) in zip(axes, COHORTS):
        d = cohort_data.get(key)
        ax.set_facecolor('white')
        if d is None:
            # The other cohort's job hasn't produced an h5mu yet -- a plain,
            # honest placeholder rather than leaving the panel blank with no
            # explanation.
            ax.text(0.5, 0.5, 'not yet available', ha='center', va='center',
                     fontsize=11, color='#999999', transform=ax.transAxes)
            ax.set_title(label, fontsize=12, pad=10)
            ax.axis('off')
            continue

        ax.scatter(d['bg_xy'][:, 0], d['bg_xy'][:, 1], c='#d3d3d3', s=0.3,
                   linewidths=0, rasterized=True, zorder=1)

        if ck in d['contrast_data']:
            cd = d['contrast_data'][ck]
            sig = cd['SpatialFDR'] < ALPHA
            n_sig = int(sig.sum())
            if n_sig > 0:
                # Only the significant subset ever gets drawn as coloured
                # points -- see the module docstring for why that's the right
                # call for a real result, unlike the Merged_v3 cosmetic plots.
                xy = d['nhood_xy'][sig]
                logfc = cd['logFC'][sig]
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=logfc, cmap=CMAP, norm=norm,
                                s=22, alpha=0.9, linewidths=0.4, edgecolors='black', zorder=2)
                sc_for_cbar = sc
            else:
                # A genuine null for this cohort/contrast (e.g. Pineda2024's
                # C9orf72_vs_Control) is stated outright rather than left as an
                # empty, ambiguous-looking panel.
                ax.text(0.5, 0.06, 'no significant neighbourhoods', ha='center', va='bottom',
                         fontsize=9.5, color='#888888', transform=ax.transAxes, style='italic')
        else:
            ax.text(0.5, 0.06, 'contrast not available', ha='center', va='bottom',
                     fontsize=9.5, color='#888888', transform=ax.transAxes, style='italic')

        ax.set_title(label, fontsize=12, pad=10)
        ax.axis('off')

    if sc_for_cbar is not None:
        # Colourbar anchored to whichever scatter actually got drawn -- if both
        # panels came back null, there's no colour mapping to show a bar for,
        # so the else-branch below computes a title-centring midpoint straight
        # from the two panels' own positions instead.
        right_pos = axes[1].get_position()
        cax = fig.add_axes([right_pos.x1 + 0.015, right_pos.y0 + right_pos.height * 0.2,
                            0.018, right_pos.height * 0.6])
        cb = fig.colorbar(sc_for_cbar, cax=cax)
        cb.set_label('log Fold Change', fontsize=11)
        left_pos = axes[0].get_position()
        mid_x = (left_pos.x0 + cax.get_position().x1) / 2
    else:
        left_pos = axes[0].get_position()
        right_pos = axes[1].get_position()
        mid_x = (left_pos.x0 + right_pos.x1) / 2

    fig.text(mid_x, 0.97, 'Neighbourhood log-Fold Change', ha='center', va='top',
              fontsize=16, fontweight='bold')
    fig.text(mid_x, 0.905, clabel, ha='center', va='top', fontsize=12.5)

    out = OUT_DIR / f'Milo_logFC_SplitRun_sig_only_{ck}.png'
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out}')

print('\nDone.')
