#!/usr/bin/env Rscript
# ============================================================
# MiloR workflow restricted to X_scVI latent space
# ============================================================

.libPaths(c("~/Rlibs", .libPaths()))
suppressPackageStartupMessages({
  library(miloR)
  library(SingleCellExperiment)
  library(scater)
  library(scran)
  library(dplyr)
  library(patchwork)
  library(zellkonverter)
  library(BiocParallel)
  library(Matrix)
  library(ggplot2)
})

options(Matrix.warnDeprecatedCoerce = FALSE)

# ---------- Config ----------
in_file <- "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/PBMC/Curated/PBMC_scVI_HVG_compat.h5ad"
out_dir  <- "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/PBMC/Milo/MiloR_results_XscVI"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

k_neighbors <- 30
prop_sample <- 0.10
alpha_fdr   <- 0.10

MAKE_PLOTS    <- FALSE
EXPORT_H5AD   <- FALSE
EXPORT_MEMBER <- TRUE

STATUS_COL <- "cell_type"
CT_COL     <- "predicted.celltype.l2"
REDUCED_DIM <- "X_scVI"

n_cores <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "16")))
if (is.na(n_cores) || n_cores < 1) n_cores <- 16
BPPARAM <- tryCatch(MulticoreParam(workers = n_cores, progressbar = TRUE),
                    error = function(e) SerialParam(progressbar = TRUE))

CONTRASTS <- list(
  Sporadic_vs_Healthy = c("Sporadic", "Healthy"),
  c9orf72_vs_Healthy  = c("c9orf72", "Healthy"),
  Sporadic_vs_c9orf72 = c("Sporadic", "c9orf72")
)

# ---------- Helpers ----------
fail     <- function(...) stop(sprintf(...), call. = FALSE)
norm_chr <- function(x) trimws(as.character(x))

write_membership <- function(milo.obj, tag = "global") {
  mat <- nhoods(milo.obj)
  Matrix::writeMM(mat, file.path(out_dir, paste0("nhoods_", tag, ".mtx")))
  write.table(rownames(mat), file = file.path(out_dir, paste0("cells_", tag, ".txt")),
              quote = FALSE, row.names = FALSE, col.names = FALSE)
  nh_ids <- colnames(mat); if (is.null(nh_ids)) nh_ids <- seq_len(ncol(mat)) - 1L
  write.table(nh_ids, file = file.path(out_dir, paste0("nhood_ids_", tag, ".txt")),
              quote = FALSE, row.names = FALSE, col.names = FALSE)
  cd <- as.data.frame(colData(milo.obj))
  if (CT_COL %in% colnames(cd)) {
    ct <- norm_chr(cd[[CT_COL]]); names(ct) <- rownames(cd); ct <- ct[rownames(mat)]
    write.table(ct, file = file.path(out_dir, paste0("celltype_", tag, ".txt")),
                quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
  if ("orig.ident" %in% colnames(cd)) {
    oi <- norm_chr(cd$orig.ident); oi <- oi[rownames(mat)]
    write.table(oi, file = file.path(out_dir, paste0("orig_ident_", tag, ".txt")),
                quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
  if ("status" %in% colnames(cd)) {
    st <- norm_chr(cd$status); st <- st[rownames(mat)]
    write.table(st, file = file.path(out_dir, paste0("status_", tag, ".txt")),
                quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
  message("Wrote membership files with tag '", tag, "'.")
}

align_design <- function(design_full, counts_cols, grpA, grpB) {
  d <- design_full |> dplyr::filter(Condition %in% c(grpA, grpB))
  if (nrow(d) == 0) fail("No samples in groups %s or %s.", grpA, grpB)
  keep <- counts_cols[counts_cols %in% d$Sample]
  if (length(keep) == 0) fail("No overlapping samples between counts and design.")
  cond_vec <- d$Condition[match(keep, d$Sample)]
  if (any(is.na(cond_vec))) fail("NA conditions after matching.")
  data.frame(Condition = factor(cond_vec, levels = c(grpB, grpA)),
             row.names = keep, check.names = FALSE)
}

# ---------- Paths ----------
pre_rds <- file.path(out_dir, "milo_preDA.rds")

# ---------- Build ----------
options(zellkonverter.useAnnDataIO = TRUE)
if (file.exists(pre_rds)) {
  message("Resuming from checkpoint: ", pre_rds)
  milo.obj <- readRDS(pre_rds)
} else {
  message("Reading single-cell AnnData from: ", in_file)
  sce <- readH5AD(in_file)
  if (!("orig.ident" %in% colnames(colData(sce)))) fail("Missing 'orig.ident'")
  if (!(STATUS_COL %in% colnames(colData(sce))))   fail("Missing '", STATUS_COL, "'")
  if (!(REDUCED_DIM %in% reducedDimNames(sce)))    fail("Missing '", REDUCED_DIM, "' reduced dim")

  colData(sce)$orig.ident <- norm_chr(colData(sce)$orig.ident)
  colData(sce)$status     <- norm_chr(colData(sce)[[STATUS_COL]])
  if (CT_COL %in% colnames(colData(sce))) colData(sce)$celltype <- norm_chr(colData(sce)[[CT_COL]])

  message("Using reduced.dim='", REDUCED_DIM, "'")
  n_comp <- ncol(reducedDim(sce, REDUCED_DIM))
  if (is.null(n_comp) || n_comp < 1) fail("0 components in '", REDUCED_DIM, "'")
  d_use <- min(n_comp, 30)

  if (!"logcounts" %in% assayNames(sce)) {
    if ("counts" %in% assayNames(sce)) logcounts(sce) <- log1p(counts(sce))
    else if ("X" %in% assayNames(sce)) assay(sce, "logcounts") <- log1p(assay(sce, "X"))
    else fail("No usable assay found.")
  }

  milo.obj <- Milo(sce)
  message("Building graph from ", REDUCED_DIM)
  milo.obj <- buildGraph(milo.obj, k = k_neighbors, d = d_use, reduced.dim = REDUCED_DIM, BPPARAM = BPPARAM)
  message("Making neighbourhoods")
  milo.obj <- makeNhoods(milo.obj, prop = prop_sample, k = k_neighbors, d = d_use, refined = TRUE)
  milo.obj <- countCells(milo.obj, meta.data = as.data.frame(colData(milo.obj)), samples = "orig.ident")
  milo.obj <- calcNhoodDistance(milo.obj, d = d_use)
  saveRDS(milo.obj, file = pre_rds)
}

if (EXPORT_MEMBER) write_membership(milo.obj, tag = "XscVI")

meta.df <- as.data.frame(colData(milo.obj))
design_full <- meta.df |> dplyr::select(Sample = orig.ident, Condition = status) |> dplyr::distinct()
counts_cols <- colnames(nhoodCounts(milo.obj))

for (label in names(CONTRASTS)) {
  grpA <- CONTRASTS[[label]][1]; grpB <- CONTRASTS[[label]][2]
  message("\n=== DA: ", label, " ===")
  design.df <- align_design(design_full, counts_cols, grpA, grpB)
  da.res <- testNhoods(milo.obj, design = ~ Condition, design.df = design.df)
  sig <- with(da.res, SpatialFDR < alpha_fdr)
  up <- sum(sig & da.res$logFC > 0, na.rm = TRUE)
  down <- sum(sig & da.res$logFC < 0, na.rm = TRUE)
  message("Significant DA: ", sum(sig), " (up=", up, ", down=", down, ")")
  write.table(as.data.frame(da.res),
              file = file.path(out_dir, paste0("DA_results_", label, ".tsv")),
              sep = "\t", quote = FALSE, row.names = TRUE)
}

sink(file.path(out_dir, "sessionInfo.txt"))
cat("===== SESSION INFO =====\n"); print(sessionInfo()); sink()
