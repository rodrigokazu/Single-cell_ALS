#!/usr/bin/env python3
"""Plot Milo logFC for each contrast onto Jingtian's tSNE coordinates, split by
cohort. v1 of this script incorrectly overlaid both cohorts on one shared canvas;
Jingtian's ALS(JCK) and Pineda2024(Kellis) tSNEs are two INDEPENDENTLY computed
runs with unrelated coordinate scales and orientation (ALS spans roughly +/-30,
Pineda roughly +/-55) -- overlaying them produces a geometrically meaningless
picture, confirmed by direct inspection. This version plots each cohort on its
own native coordinate system, side by side, which is the only valid way to show
this. INTERNAL VIEWING ONLY -- the underlying Merged_v3 neighbourhood graph and
DA statistics are unchanged and remain void (0/62,952 significant nhoods in
every contrast, non-converged size factors at graph-build time).
"""
# ==============================================================================
# WHAT THIS IS
#   A cosmetic re-plot, not a re-analysis. It takes the Milo results already
#   sitting in Merged_v3/Milo_v3.h5mu (logFC and SpatialFDR per neighbourhood,
#   per contrast) and draws them on Jingtian's real tSNE coordinates instead of
#   that h5mu's own (independently found row-permuted) X_tsne. Nothing about the
#   neighbourhood graph or the differential-abundance statistics changes; only
#   the plotting coordinate does.
#
# HOW v1 GOT IT WRONG, AND HOW THAT WAS CAUGHT
#   The first version of this script plotted every cohort's neighbourhoods on
#   one shared pair of axes, using build_jingtian_coords_fulldataset.py's
#   per-cell remap directly. That remap itself was correct -- barcode matching
#   had already been confirmed at 100% -- but Jingtian's ALS and Pineda2024
#   tSNEs are two separately computed runs with no shared coordinate frame, so
#   putting both cohorts' points on one canvas mixed two unrelated spatial
#   scales into a single picture. Rodrigo caught this by eye: the shapes didn't
#   match a known-good single-cohort reference plot. Splitting the diagnostic
#   coordinate array by cohort and plotting each cohort alone reproduced that
#   reference exactly, which is what confirmed the bug was in the plotting
#   choice, not in the underlying barcode match.
#
# WHY THIS STILL DOESN'T RESCUE THE MANUSCRIPT'S ED FIG. 1B
#   Better axes do not fix a broken statistical result underneath them. The
#   Merged_v3 run this script reads from has its own X_pca_corrected
#   row-permuted at the point the neighbourhood graph was built, and its
#   pyDESeq2 size factors never converged -- neither defect has anything to do
#   with what the result gets plotted on. Every panel this script produces
#   will faithfully show 0 significant neighbourhoods in every contrast,
#   because that is what is actually in the data; it will just do so on
#   coordinates that finally line up with the rest of the paper's figures.
#   This script exists for internal viewing only, at Rodrigo's explicit
#   request, and was never intended to answer the reviewer's ED Fig. 1b
#   comment on its own.
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

# The already-void full-merged Milo result -- read for its logFC/SpatialFDR
# only, never for its own X_tsne (that's exactly what's being replaced).
H5MU = '/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Milo/MiloP_results/Merged_v3/Milo_v3.h5mu'
# Output of build_jingtian_coords_fulldataset.py -- the barcode-matched,
# per-cohort Jingtian coordinates this script plots instead.
COORDS_NPZ = '/home/rodrigok/tmp_check/jingtian_coords_fulldataset.npz'
OUT_DIR = Path('/home/rodrigok/tmp_check/jingtian_coord_plots_v2')
OUT_DIR.mkdir(exist_ok=True, parents=True)

ALPHA   = 0.1
CMAP    = 'RdBu_r'
# Only three of the original pipeline's four contrasts -- C9orf72_vs_Sporadic is
# intentionally left out of this particular plotting pass; nothing deeper than
# that, the other three were the ones actually needed for this viewing.
CONTRASTS = [
    ('ALS_vs_Control',     'ALS vs Control'),
    ('C9orf72_vs_Control', 'C9orf72 vs Control'),
    ('Sporadic_vs_Control','Sporadic vs Control'),
]
COHORTS = [('ALS', 'Our dataset'), ('Pineda2024', 'Pineda et al')]

print('Loading Jingtian-sourced coordinates + cohort labels ...')
npz = np.load(COORDS_NPZ, allow_pickle=True)
tsne = npz['xy']
cell_idx_npz = npz['cell_idx']
study = npz['study']

print('Reading h5mu ...')
with h5py.File(H5MU, 'r') as f:
    cell_idx   = f['mod/rna/obs/_index'].asstr()[:]
    ic         = f['mod/milo/var/index_cell'].asstr()[:]
    contrast_data = {}
    for ck, _ in CONTRASTS:
        contrast_data[ck] = {
            'logFC':      f[f'mod/milo/var/logFC__{ck}'][:],
            'SpatialFDR': f[f'mod/milo/var/SpatialFDR__{ck}'][:],
        }

# Guards against a silent, hard-to-notice bug: if the .npz was ever rebuilt from
# a different h5mu snapshot than the one being read here, the two cell orders
# could drift apart and every downstream index-based lookup would quietly point
# at the wrong cell. This turns that into a loud failure instead.
assert np.array_equal(cell_idx, cell_idx_npz), 'cell_idx order mismatch between h5mu and npz'

# Barcode-keyed join from Milo neighbourhood index-cells to a row in the
# coordinate array -- positional order is never assumed, which is deliberate:
# a positional assumption exactly like this is what let the original Merged_v3
# corruption go undetected for as long as it did elsewhere in this project.
cell_pos   = {b: i for i, b in enumerate(cell_idx)}
nhood_rows = np.array([cell_pos.get(c, -1) for c in ic])
valid      = nhood_rows >= 0
nhood_xy_all    = tsne[nhood_rows[valid]]
nhood_study_all = study[nhood_rows[valid]]
print(f'{valid.sum():,} nhoods matched to a cell row')

for ck, label in CONTRASTS:
    logfc_all = contrast_data[ck]['logFC'][valid]
    sfdr_all  = contrast_data[ck]['SpatialFDR'][valid]
    finite_all = logfc_all[np.isfinite(logfc_all)]
    # Colour scale is fixed from the 99th percentile of |logFC| across BOTH
    # cohorts before the per-cohort loop below, so the same colour means the
    # same effect size in both panels of a given contrast's figure.
    vmax = float(np.nanpercentile(np.abs(finite_all), 99)) if len(finite_all) else 1.0
    if vmax == 0: vmax = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.8), facecolor='white')
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.03, right=0.90, wspace=0.06)
    sc_for_cbar = None

    for ax, (cohort_key, cohort_label) in zip(axes, COHORTS):
        # This is the fix, in one line: each cohort's background AND
        # neighbourhoods are masked and plotted entirely separately, on its own
        # axes object, using only that cohort's own coordinates. Nothing from
        # one cohort ever appears on the other's axes -- see the module
        # docstring for why v1's shared canvas was wrong.
        cmask = nhood_study_all == cohort_key
        bg_mask = study == cohort_key
        bg_xy = tsne[bg_mask]
        bg_finite = ~np.isnan(bg_xy).any(axis=1)

        xy = nhood_xy_all[cmask]
        finite_xy = ~np.isnan(xy).any(axis=1)
        xy = xy[finite_xy]
        logfc = logfc_all[cmask][finite_xy]
        sfdr  = sfdr_all[cmask][finite_xy]
        sig   = sfdr < ALPHA

        ax.set_facecolor('white')
        # Faint grey backdrop of every cell in this cohort, drawn first so the
        # coloured neighbourhood points sit visibly on top of it.
        ax.scatter(bg_xy[bg_finite, 0], bg_xy[bg_finite, 1],
                   c='#d3d3d3', s=0.3, linewidths=0, rasterized=True, zorder=1)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=logfc, cmap=CMAP, norm=norm,
                         s=18, alpha=0.85, linewidths=0, zorder=2)
        if sig.any():
            # Black ring around any neighbourhood that actually clears
            # SpatialFDR < ALPHA. In the Merged_v3 data this predictably never
            # fires -- 0 of 62,952 neighbourhoods are significant in any
            # contrast -- but the check stays in so the plot would tell the
            # truth immediately if that ever changed.
            ax.scatter(xy[sig, 0], xy[sig, 1], facecolors='none', edgecolors='black',
                       s=22, linewidths=0.6, zorder=3)
        ax.set_title(cohort_label, fontsize=12, pad=10)
        ax.axis('off')
        sc_for_cbar = sc

    # Colourbar placed as its own small axes flush against the right panel,
    # rather than matplotlib's automatic placement, specifically to avoid
    # leaving a dead strip of empty canvas between the plot and the bar.
    right_pos = axes[1].get_position()
    cax = fig.add_axes([right_pos.x1 + 0.015, right_pos.y0 + right_pos.height * 0.2,
                        0.018, right_pos.height * 0.6])
    cb = fig.colorbar(sc_for_cbar, cax=cax)
    cb.set_label('log Fold Change', fontsize=11)

    # Title/subtitle x-position is computed from the actual left edge of panel
    # one and the actual right edge of the colourbar, not a flat 0.5 of the
    # figure -- centring on the true content width rather than the full canvas
    # is what keeps the title looking centred after bbox_inches='tight' trims
    # any leftover margin at save time.
    left_pos = axes[0].get_position()
    mid_x = (left_pos.x0 + cax.get_position().x1) / 2
    fig.text(mid_x, 0.97, 'Neighbourhood log-Fold Change', ha='center', va='top',
              fontsize=16, fontweight='bold')
    fig.text(mid_x, 0.905, label, ha='center', va='top', fontsize=12.5)

    out = OUT_DIR / f'Milo_logFC_JingtianTSNE_splitcohort_{ck}.png'
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out}')

print('\nDone. Reminder: 0 significant nhoods in the original Merged_v3 run in every')
print('contrast; splitting by cohort fixes the geometry but not the underlying DA result.')
