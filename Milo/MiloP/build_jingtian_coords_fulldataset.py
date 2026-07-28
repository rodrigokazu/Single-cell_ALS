"""Build a barcode-matched Jingtian tSNE coordinate for every one of the 778,330
cells in the manuscript's full-merged Milo h5mu (Merged_v3/Milo_v3.h5mu), by
looking each cell up in Jingtian's own real per-cohort output files rather than
using that h5mu's own (independently found to be row-permuted) X_tsne.
"""
# ==============================================================================
# WHAT THIS IS
#   A one-off coordinate-remapping step, not part of any statistical analysis.
#   It produces a lookup array (saved as an .npz) mapping every cell in the
#   full-merged h5mu to an (x, y) position from Jingtian's own tSNE, used
#   downstream purely for PLOTTING -- specifically the cosmetic, internal-viewing-
#   only re-plot of the already-void Merged_v3 Milo result on a trustworthy
#   coordinate system (see milo_tsne_logfc_plot_jingtian_v2.py).
#
# WHY IT'S A PLAIN PYTHON LOOP OVER 778,330 CELLS INSTEAD OF A VECTORISED JOIN
#   Each cell's source cohort (study == "ALS" or "Pineda2024") decides which of
#   two entirely separate lookup dictionaries to use, and a vectorised approach
#   would need to build and discard a lot of intermediate masks for what is, in
#   practice, a single linear pass with two dict.get() calls. On this cluster's
#   login node the loop finished in well under a minute, so there was no reason
#   to reach for anything cleverer.
#
# WHY TWO SEPARATE LOOKUPS, NOT ONE
#   Jingtian computed the ALS(JCK) and Pineda2024(Kellis) embeddings completely
#   independently -- two unrelated tSNE runs, each only anchored to a shared
#   MSMO reference panel, never corrected against each other. A cell's cohort
#   membership is not incidental bookkeeping here; it determines which of two
#   unrelated coordinate systems its position even means anything in. Mixing
#   them into a single canvas is exactly the mistake a later version of the
#   downstream plotting script made and had to be corrected for (see the
#   overlay-bug note in milo_tsne_logfc_plot_jingtian_v2.py's own history) --
#   this script itself was never wrong, but it's worth recording the same
#   caveat here, since anyone re-using its output needs to keep the two
#   cohorts' coordinates in separate plot panels, never one shared scatter.
#
# THE BARCODE-FORMAT ASYMMETRY
#   Rodrigo's own h5mu already stores ALS-origin barcodes in the concatenated
#   "sample_barcode" form (matching Jingtian's ALS file once sample + bare
#   barcode are joined below) and Pineda-origin barcodes in a suffix form that
#   matches Jingtian's Pineda file's native index with no transformation needed
#   at all. Confirmed to reach 100% matched, 0 unmatched, for both cohorts,
#   across all 778,330 cells, before this output was trusted for anything.
#
# WHAT THIS INCIDENTALLY PROVED
#   A downstream, more targeted test (matching obsm['X_pca_corrected'] itself,
#   not just this tSNE, against Jingtian's real files for 500 sampled cells per
#   cohort) came back bit-exact, max absolute difference 0.0, for both cohorts.
#   Because a bit-exact match on a stochastic embedding method is only possible
#   if its input was carried over unchanged rather than recomputed, that result
#   confirms directly, not just by inference, that the "merged" embedding this
#   project's Milo pipeline reads is a byte-for-byte concatenation of Jingtian's
#   two independently-computed per-cohort spaces -- never jointly batch-
#   corrected against each other at any point.
# ==============================================================================
# Comments added 28 July 2026. Executable logic is byte-identical to the
# version that ran; a pristine copy sits in _verbatim/, and every non-docstring
# statement was confirmed unchanged with verify_comments_only.py before this
# copy shipped.
# ==============================================================================
import h5py
import numpy as np

# The full-merged Milo h5mu this script is remapping coordinates FOR -- this is
# read only for its cell index and cohort ("study") label, never its own
# (already known-corrupted) X_tsne.
H5MU = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Milo/MiloP_results/Merged_v3/Milo_v3.h5mu"
# Jingtian's own real per-cohort output files -- the actual location, not the
# (nonexistent-on-this-cluster) path his own scripts read from.
JCK_FILE = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Jingtian/MSMO_ALS_RNA_merged_lr60.h5ad"
KEL_FILE = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/ALS_multiome/Jingtian/MSMO_Pineda_RNA_merged.h5ad"
OUT_NPZ = "/home/rodrigok/tmp_check/jingtian_coords_fulldataset.npz"

def cat_col(h, name):
    # h5py stores a pandas categorical as a group with categories/codes rather
    # than a flat string array; this handles both shapes transparently.
    g = h["obs"][name]
    if hasattr(g, "keys"):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
        codes = g["codes"][:]
        return cats[codes]
    vals = g[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in vals])

def read_index(h):
    idx = h["obs"]["_index"][:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in idx])

print("Reading h5mu cell index + study column ...")
hm = h5py.File(H5MU, "r")
# Index-only, no X/layers/obsm touched here -- this file is large and its own
# tSNE is exactly what's being replaced, so there's nothing else worth reading.
h5mu_idx = hm["mod/rna/obs/_index"].asstr()[:]
study_grp = hm["mod/rna/obs/study"]
if hasattr(study_grp, "keys"):
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in study_grp["categories"][:]])
    codes = study_grp["codes"][:]
    study = cats[codes]
else:
    vals = study_grp[:]
    study = np.array([x.decode() if isinstance(x, bytes) else x for x in vals])
print(f"  {len(h5mu_idx):,} cells; study counts: {dict(zip(*np.unique(study, return_counts=True)))}")
hm.close()

print("Reading Jingtian's ALS/JCK file ...")
jh = h5py.File(JCK_FILE, "r")
jck_bare = read_index(jh)
jck_sample = cat_col(jh, "sample")
# Jingtian's ALS file keeps sample ID and bare 16bp barcode as two separate
# columns; concatenating them here is what makes this key comparable to
# Rodrigo's own h5mu barcodes, which already store the joined form.
jck_key = np.array([f"{s}_{b}" for s, b in zip(jck_sample, jck_bare)])
jck_tsne = jh["obsm"]["pc40seuratcc30_tsne"][:]
jh.close()
jck_lookup = {k: i for i, k in enumerate(jck_key)}
print(f"  {len(jck_key):,} JCK cells with tsne")

print("Reading Jingtian's Pineda/Kellis file ...")
kh = h5py.File(KEL_FILE, "r")
kel_key = read_index(kh)  # native format already matches h5mu Pineda2024 barcodes
kel_tsne = kh["obsm"]["pc40seuratcc30_tsne"][:]
kh.close()
kel_lookup = {k: i for i, k in enumerate(kel_key)}
print(f"  {len(kel_key):,} Kellis cells with tsne")

print("Building remapped coordinate array for all h5mu cells ...")
n = len(h5mu_idx)
# NaN, not zero, for anything unmatched -- a real (0, 0) coordinate would plot
# as a visible point; NaN simply won't be drawn by matplotlib, which is the
# correct behaviour for a cell this remap couldn't place.
new_xy = np.full((n, 2), np.nan, dtype=np.float64)
n_matched_als = 0
n_matched_pin = 0
n_unmatched_als = 0
n_unmatched_pin = 0

for i in range(n):
    b = h5mu_idx[i]
    s = study[i]
    # Cohort membership picks which of the two independent lookups applies --
    # never both, and never a shared one. See the module docstring for why
    # that split is load-bearing, not a convenience.
    if s == "ALS":
        j = jck_lookup.get(b)
        if j is not None:
            new_xy[i] = jck_tsne[j]
            n_matched_als += 1
        else:
            n_unmatched_als += 1
    else:
        j = kel_lookup.get(b)
        if j is not None:
            new_xy[i] = kel_tsne[j]
            n_matched_pin += 1
        else:
            n_unmatched_pin += 1

print(f"ALS:    matched {n_matched_als:,} / unmatched {n_unmatched_als:,}")
print(f"Pineda: matched {n_matched_pin:,} / unmatched {n_unmatched_pin:,}")
print(f"Total matched: {n_matched_als + n_matched_pin:,} / {n:,} "
      f"({100*(n_matched_als+n_matched_pin)/n:.2f}%)")

# study is saved alongside xy/cell_idx so downstream plotting scripts can split
# by cohort without re-deriving it -- see milo_tsne_logfc_plot_jingtian_v2.py,
# which reads exactly this file.
np.savez(OUT_NPZ, xy=new_xy, cell_idx=h5mu_idx, study=study)
print(f"Saved -> {OUT_NPZ}")
