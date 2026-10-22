# export_sce.R
# -----------------------------------------------------------------------------
# Convert a SingleCellExperiment (.rds, Hemberg-lab scRNA-seq collection format)
# into plain files that Python/AnnData can read without any R bridge:
#
#   <out_dir>/<assay>.mtx   sparse MatrixMarket, GENES x CELLS (R orientation)
#   <out_dir>/obs.csv       colData  -> per-cell metadata (cell labels, donor, QC)
#   <out_dir>/var.csv       rowData  -> per-gene metadata
#
# Why plain files instead of zellkonverter / anndata2ri?
#   * zellkonverter is not installed and pulls a whole conda env via basilisk;
#     MatrixMarket + CSV are dependency-free and human-inspectable.
#   * The export is done ONCE; Python then caches it as .h5ad (fast to reload).
#
# Every assay present in the object is exported (Baron: counts + logcounts,
# Camp: logcounts only), so that the Python side can decide which one to use
# and we never silently lose the raw counts.
#
# Usage: Rscript scripts/export_sce.R <input.rds> <out_dir>
# -----------------------------------------------------------------------------
suppressMessages({
  library(SingleCellExperiment)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
in_rds  <- args[1]
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sce <- readRDS(in_rds)
cat("Loaded", in_rds, ":", nrow(sce), "genes x", ncol(sce), "cells\n")

# Sanity checks before writing: names must be unique, otherwise the Python
# side would have ambiguous indices.
stopifnot(!anyDuplicated(rownames(sce)), !anyDuplicated(colnames(sce)))

for (a in assayNames(sce)) {
  m <- assay(sce, a)
  stopifnot(!anyNA(m))                      # no missing values in the matrix
  m <- as(m, "CsparseMatrix")               # dense -> sparse (scRNA-seq is ~90% zeros)
  writeMM(m, file.path(out_dir, paste0(a, ".mtx")))
  cat("  wrote assay", a, "with", length(m@x), "non-zero entries\n")
}

obs <- as.data.frame(colData(sce))
obs <- cbind(cell_id = colnames(sce), obs)
write.csv(obs, file.path(out_dir, "obs.csv"), row.names = FALSE)

var <- as.data.frame(rowData(sce))
var <- cbind(gene_id = rownames(sce), var)
write.csv(var, file.path(out_dir, "var.csv"), row.names = FALSE)
cat("Done ->", out_dir, "\n")
