#!/usr/bin/env python3
"""Aggregate the split-cohort Milo neighbourhood-level results up to cell-type
resolution, using each neighbourhood's majority annotation as a proxy cell-type
label, and ship them as one comparison workbook, the same shape as the scCODA
3-way spreadsheet already delivered for this project.
"""
# ==============================================================================
# WHAT THIS IS
#   The answer to JCK's separate ask (distinct from the reviewer's own Milo
#   comment) for a Milo counterpart to the three-way scCODA comparison, built
#   the way project memory said it would have to be: Milo tests neighbourhoods,
#   not cell types, and a like-for-like neighbourhood-level three-way comparison
#   returns 0 significant neighbourhoods everywhere by sheer power arithmetic.
#   The fix is to report Milo at the SAME resolution scCODA uses -- cell type --
#   by rolling neighbourhoods up via their majority annotation. That is exactly
#   what this script does, once real per-cohort results existed to roll up.
#
# THE CAVEAT THAT MATTERS MOST, REPEATED IN THE WORKBOOK'S OWN README SHEET
#   Milo neighbourhoods OVERLAP by construction: a single cell can belong to
#   several neighbourhoods, because neighbourhoods come from a KNN graph, not
#   from a partition. Grouping neighbourhoods by their majority cell type is
#   therefore a DESCRIPTIVE roll-up of many redundant, overlapping tests, not a
#   formal per-cell-type statistical test the way scCODA's is, since scCODA
#   tests each cell type exactly once against an exclusive partition of cells.
#   The output columns are named and the README sheet is worded specifically to
#   keep that distinction visible: frac_sig and mean_logFC are read as
#   descriptive signal density, never as a cell-type-level p-value, because no
#   such p-value was computed and reporting one would overstate what a Milo
#   neighbourhood re-aggregation can actually support.
#
# WHY THERE IS NO "INTEGRATED" (POOLED) ARM
#   The scCODA three-way workbook has three arms: Integrated, Multiome-alone,
#   Pineda2024-alone. This workbook only has two, because the natural
#   Integrated arm for Milo -- the full-merged run, Merged_v3 -- is
#   independently void: its own X_pca_corrected was row-permuted at the point
#   the neighbourhood graph got built, and its pyDESeq2 size factors never
#   converged. Reporting it alongside two valid per-cohort results would put a
#   broken number next to two trustworthy ones with no visual distinction
#   between them, so it is left out entirely rather than included with an
#   asterisk. Two honest arms beat three where one is fiction.
#
# WHY THIS SCRIPT TOLERATES A MISSING COHORT
#   Written and first validated against Pineda2024 alone, while the ALS/JCK
#   job was still running on the cluster. load_dataset() returns None for any
#   h5mu that doesn't exist yet, and main() reports that cohort as "NOT
#   AVAILABLE" and carries on with whatever it has, rather than failing
#   outright. Re-running this exact script, unmodified, after the second
#   cohort's job finishes is what produces the complete two-way workbook --
#   there is no cohort-specific branch to edit or uncomment.
#
# WHAT IT PRODUCES
#   One .xlsx with four sheets: README (this explanation, written into the
#   file itself so it travels with it), Full_results_long (one row per
#   dataset x contrast x cell_type, with neighbourhood counts, significant
#   counts, fraction significant, direction split, mean/median logFC, mean
#   neighbourhood purity and mean neighbourhood size), Comparison_wide (the
#   same fraction-significant numbers reshaped so both datasets sit side by
#   side per contrast x cell_type, for quick eyeballing), and Discrepancies
#   (the same wide table, restricted to rows where both datasets have data and
#   sorted by how far apart their significant fractions are).
# ==============================================================================
# Comments added 28 July 2026. Executable logic is byte-identical to the
# version that ran; a pristine copy sits in _verbatim/, and every non-docstring
# statement was confirmed unchanged with verify_comments_only.py before this
# copy shipped.
# ==============================================================================
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sp

ALPHA = 0.10
CONTRASTS = ['ALS_vs_Control', 'C9orf72_vs_Control', 'Sporadic_vs_Control', 'C9orf72_vs_Sporadic']
# The two valid per-cohort split-run outputs (milo_split_cohort_run.py). No
# third "Integrated" entry -- see the module docstring for why that arm is
# deliberately absent rather than included as a known-void number.
DATASETS = {
    'OurCohort': '/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Milo/MiloP_results/JingtianSplit_ALS/Milo_ALS.h5mu',
    'Pineda2024': '/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Milo/MiloP_results/JingtianSplit_Pineda2024/Milo_Pineda2024.h5mu',
}
OUT_XLSX = '/home/rodrigok/tmp_check/Milo_celltype_aggregated_2way.xlsx'


def nhood_sizes_from_h5mu(f):
    # milopy stores the cell x neighbourhood membership matrix as a CSR sparse
    # array; a neighbourhood's size is just the column sum of how many cells
    # belong to it. Reconstructed directly from the raw data/indices/indptr
    # rather than through anndata, since only this one derived quantity is
    # needed and the full object carries far more than that.
    nh = f['mod/rna/obsm/nhoods']
    data = nh['data'][:]
    indices = nh['indices'][:]
    indptr = nh['indptr'][:]
    shape = tuple(nh.attrs['shape'])
    mat = sp.csr_matrix((data, indices, indptr), shape=shape)
    return np.asarray(mat.sum(axis=0)).ravel()


def load_dataset(path):
    # Missing file -> None, not an exception. This is what lets main() report
    # "NOT AVAILABLE" for a cohort whose job hasn't finished instead of crashing
    # the whole workbook build.
    if not Path(path).exists():
        return None
    f = h5py.File(path, 'r')
    annot_grp = f['mod/milo/var/nhood_annotation']
    if hasattr(annot_grp, 'keys'):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in annot_grp['categories'][:]])
        codes = annot_grp['codes'][:]
        annotation = cats[codes]
    else:
        annotation = np.array([x.decode() if isinstance(x, bytes) else x for x in annot_grp[:]])
    purity = f['mod/milo/var/nhood_purity'][:]
    sizes = nhood_sizes_from_h5mu(f)

    contrast_data = {}
    for ck in CONTRASTS:
        lfc_key = f'mod/milo/var/logFC__{ck}'
        fdr_key = f'mod/milo/var/SpatialFDR__{ck}'
        # A contrast that hasn't been fitted (or failed) simply isn't in the
        # h5mu -- treated as absent for this dataset, not as an error.
        if lfc_key in f and fdr_key in f:
            contrast_data[ck] = {'logFC': f[lfc_key][:], 'SpatialFDR': f[fdr_key][:]}
    f.close()
    return {'annotation': annotation, 'purity': purity, 'sizes': sizes, 'contrast_data': contrast_data}


def aggregate(dataset_name, d):
    # This is the actual roll-up: every neighbourhood already carries a
    # majority-cell-type label (assigned back in milo_split_cohort_run.py), so
    # grouping by that label and summarising logFC/significance within each
    # group is the whole of the "aggregate to cell-type resolution" step. No
    # new statistical test is fitted here -- see the module docstring's caveat
    # about why that would overstate what overlapping neighbourhoods support.
    rows = []
    for ck in CONTRASTS:
        if ck not in d['contrast_data']:
            continue
        logfc = d['contrast_data'][ck]['logFC']
        sfdr = d['contrast_data'][ck]['SpatialFDR']
        sig = sfdr < ALPHA
        df = pd.DataFrame({
            'cell_type': d['annotation'], 'logFC': logfc, 'sig': sig,
            'purity': d['purity'], 'size': d['sizes'],
        })
        for ct, g in df.groupby('cell_type'):
            n_total = len(g)
            n_sig = int(g['sig'].sum())
            sig_g = g[g['sig']]
            n_up = int((sig_g['logFC'] > 0).sum())
            n_down = int((sig_g['logFC'] < 0).sum())
            rows.append({
                'dataset': dataset_name, 'contrast': ck, 'cell_type': ct,
                'n_nhoods': n_total,
                'n_sig': n_sig,
                'frac_sig': n_sig / n_total if n_total else np.nan,
                'n_up_in_A': n_up, 'n_down_in_A': n_down,
                'mean_logFC_all': g['logFC'].mean(),
                'median_logFC_all': g['logFC'].median(),
                'mean_logFC_sig': sig_g['logFC'].mean() if n_sig else np.nan,
                'median_logFC_sig': sig_g['logFC'].median() if n_sig else np.nan,
                # Diagnostic columns, not part of the finding itself: mean
                # purity flags cell types whose neighbourhoods are internally
                # mixed (a low-purity cell type's frac_sig is less trustworthy
                # as a label for what's actually driving the signal), and mean
                # neighbourhood size flags cell types resting on very few
                # cells per neighbourhood.
                'mean_purity': g['purity'].mean(),
                'mean_nhood_size': g['size'].mean(),
            })
    return pd.DataFrame(rows)


def main():
    all_long = []
    missing = []
    for name, path in DATASETS.items():
        d = load_dataset(path)
        if d is None:
            missing.append(name)
            print(f'{name}: NOT AVAILABLE ({path})')
            continue
        agg = aggregate(name, d)
        all_long.append(agg)
        print(f'{name}: {len(agg)} (contrast x cell_type) rows from {len(d["annotation"]):,} neighbourhoods')

    if not all_long:
        print('Nothing available yet.')
        return

    long_df = pd.concat(all_long, ignore_index=True)
    long_df = long_df.sort_values(['contrast', 'cell_type', 'dataset']).reset_index(drop=True)

    # wide comparison: frac_sig per dataset, side by side, per contrast x cell_type
    wide = long_df.pivot_table(index=['contrast', 'cell_type'], columns='dataset',
                                values='frac_sig').reset_index()

    # discrepancies: both datasets present for that contrast x cell_type, and their
    # sig-fraction differs by a wide margin (descriptive threshold, not a formal test)
    discrepancies = None
    if set(DATASETS.keys()).issubset(set(wide.columns)):
        w = wide.dropna(subset=list(DATASETS.keys()))
        w = w.copy()
        w['abs_diff'] = (w['OurCohort'] - w['Pineda2024']).abs()
        discrepancies = w.sort_values('abs_diff', ascending=False)

    Path(OUT_XLSX).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as xl:
        # The caveats above are written into the workbook itself, not just this
        # script's comments -- anyone who opens the .xlsx without ever reading
        # the source should still see the "descriptive, not a formal test" and
        # "no Integrated arm, and here's why" notes before looking at a number.
        readme = pd.DataFrame({'Note': [
            'Milo neighbourhood-level results, aggregated to cell-type resolution using each',
            'neighbourhood\'s MAJORITY cell type (nhood_annotation) as a proxy label.',
            '',
            'This is a DESCRIPTIVE roll-up, not a formal per-cell-type statistical test.',
            'Milo neighbourhoods OVERLAP by design (a cell can belong to several',
            'neighbourhoods), so grouping by majority cell type re-uses overlapping tests --',
            'unlike scCODA, which tests each cell type once on an exclusive partition.',
            'Read frac_sig / mean_logFC as descriptive signal density, not a p-value.',
            '',
            'Datasets covered: ' + ', '.join(DATASETS.keys()),
            'Missing (not yet run): ' + (', '.join(missing) if missing else 'none'),
            '',
            'There is no "Integrated" arm here -- the joint/merged Milo run (Merged_v3) is',
            'independently void (row-permuted PCA input, non-converged size factors), so it',
            'is excluded rather than reported alongside these valid per-cohort results.',
            '',
            'Significance threshold: SpatialFDR < 0.10, matching the original pipeline.',
        ]})
        readme.to_excel(xl, sheet_name='README', index=False)
        long_df.to_excel(xl, sheet_name='Full_results_long', index=False)
        wide.to_excel(xl, sheet_name='Comparison_wide', index=False)
        if discrepancies is not None:
            discrepancies.to_excel(xl, sheet_name='Discrepancies', index=False)

    print(f'\nSaved -> {OUT_XLSX}')
    print(f'Total long rows: {len(long_df)}')

if __name__ == '__main__':
    main()
