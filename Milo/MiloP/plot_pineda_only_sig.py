#!/usr/bin/env python3
"""Single-panel significant-neighbourhoods plot for Pineda2024 alone (ALS/JCK not
yet available). Same styling as plot_split_sig_only.py, single panel instead of two."""
# ==============================================================================
# WHAT THIS IS, AND WHY IT EXISTS ALONGSIDE plot_split_sig_only.py
#   plot_split_sig_only.py always draws a two-panel figure, with a plain
#   placeholder for any cohort whose job hasn't finished yet. That's the right
#   default when the story is "the split-cohort re-analysis, both cohorts," but
#   it reads oddly for a report that is honestly and deliberately scoped to
#   Pineda2024 alone, with the other cohort's analysis not started rather than
#   merely pending -- a two-panel layout with a permanent "not yet available"
#   side implies a joint comparison that this particular deliverable was never
#   claiming to make. This script draws the single, real panel on its own,
#   sized and centred as a complete figure rather than half of one.
#
# WHAT IT PRODUCES
#   One PNG per contrast, significant neighbourhoods only (SpatialFDR < ALPHA),
#   on Pineda2024's own native tSNE. Cell types, colour scale and the
#   significant-only filtering rule are identical to plot_split_sig_only.py;
#   only the layout (one panel, not two) differs.
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
H5MU = '/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Milo/MiloP_results/JingtianSplit_Pineda2024/Milo_Pineda2024.h5mu'
OUT_DIR = Path('/home/rodrigok/tmp_check/pineda_only_plots')
OUT_DIR.mkdir(exist_ok=True, parents=True)

CONTRASTS = [
    ('ALS_vs_Control',     'ALS vs Control'),
    ('C9orf72_vs_Control', 'C9orf72 vs Control'),
    ('Sporadic_vs_Control','Sporadic vs Control'),
    ('C9orf72_vs_Sporadic','C9orf72 vs Sporadic'),
]

f = h5py.File(H5MU, 'r')
tsne = f['mod/rna/obsm/' + TSNE_KEY][:]
cell_idx = f['mod/rna/obs/_index'].asstr()[:]
index_cell = f['mod/milo/var/index_cell'].asstr()[:]
contrast_data = {}
for ck, _ in CONTRASTS:
    contrast_data[ck] = {
        'logFC': f[f'mod/milo/var/logFC__{ck}'][:],
        'SpatialFDR': f[f'mod/milo/var/SpatialFDR__{ck}'][:],
    }
f.close()

# Barcode-keyed join from neighbourhood index-cells to a tSNE row -- same
# pattern used everywhere else in this investigation, never a positional
# assumption between the milo and rna modalities.
cell_pos = {b: i for i, b in enumerate(cell_idx)}
nhood_rows = np.array([cell_pos.get(c, -1) for c in index_cell])
valid = nhood_rows >= 0
nhood_xy = tsne[nhood_rows[valid]]

for ck, clabel in CONTRASTS:
    d = {k: v[valid] for k, v in contrast_data[ck].items()}
    sig = d['SpatialFDR'] < ALPHA
    n_sig = int(sig.sum())

    fig, ax = plt.subplots(figsize=(7.6, 7.6), facecolor='white')
    fig.subplots_adjust(top=0.82, bottom=0.03, left=0.03, right=0.85)
    ax.set_facecolor('white')
    ax.scatter(tsne[:, 0], tsne[:, 1], c='#d3d3d3', s=0.3, linewidths=0, rasterized=True, zorder=1)

    if n_sig > 0:
        # Colour scale here is sized from THIS contrast's own significant
        # neighbourhoods only, unlike plot_split_sig_only.py, which shares one
        # scale across two panels. With only one panel to show, there's no
        # second cohort's values to stay comparable with.
        xy = nhood_xy[sig]
        logfc = d['logFC'][sig]
        vmax = float(np.nanpercentile(np.abs(logfc), 99))
        if vmax == 0 or not np.isfinite(vmax):
            vmax = float(np.nanmax(np.abs(logfc))) or 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=logfc, cmap=CMAP, norm=norm,
                        s=24, alpha=0.9, linewidths=0.4, edgecolors='black', zorder=2)
        right_pos = ax.get_position()
        cax = fig.add_axes([right_pos.x1 + 0.02, right_pos.y0 + right_pos.height * 0.2,
                            0.025, right_pos.height * 0.6])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label('log Fold Change', fontsize=11)
        mid_x = (right_pos.x0 + cax.get_position().x1) / 2
    else:
        # A genuine null (C9orf72_vs_Control and C9orf72_vs_Sporadic both land
        # here for Pineda2024) is stated outright rather than left as an empty,
        # ambiguous panel with no colourbar and no explanation.
        ax.text(0.5, 0.06, 'no significant neighbourhoods', ha='center', va='bottom',
                 fontsize=10, color='#888888', transform=ax.transAxes, style='italic')
        mid_x = (ax.get_position().x0 + ax.get_position().x1) / 2

    ax.axis('off')
    fig.text(mid_x, 0.97, 'Neighbourhood log-Fold Change', ha='center', va='top',
              fontsize=15, fontweight='bold')
    fig.text(mid_x, 0.905, f'{clabel}  —  Pineda et al only', ha='center', va='top', fontsize=11.5)

    out = OUT_DIR / f'Milo_logFC_PinedaOnly_sig_{ck}.png'
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'{ck}: {n_sig} sig, saved -> {out}')
