"""
build_h5ad.py
-------------------------------------------------------------------------------
Second half of the R -> Python conversion (first half: scripts/export_sce.R).

Reads the MatrixMarket/CSV export of one SingleCellExperiment and writes a
single AnnData file (.h5ad) in the scanpy convention:

    adata.X                 -> the "primary" matrix (raw counts if available,
                               otherwise whatever the authors deposited)
    adata.layers[<assay>]   -> every exported assay, kept side by side
    adata.obs               -> per-cell metadata  (CELLS are rows)
    adata.var               -> per-gene metadata  (GENES are columns)

Note the orientation flip: R/Bioconductor stores GENES x CELLS, scanpy/sklearn
store CELLS x GENES (samples are rows, features are columns). We transpose
once here so that every downstream step can treat a cell as an ML sample.

Usage: python -I scripts/build_h5ad.py <export_dir> <out.h5ad>
"""
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.io
import scipy.sparse as sp


def build(export_dir: Path, out_path: Path) -> ad.AnnData:
    obs = pd.read_csv(export_dir / "obs.csv", index_col="cell_id")
    var = pd.read_csv(export_dir / "var.csv", index_col="gene_id")
    # Index names must be plain strings for h5ad serialisation.
    obs.index = obs.index.astype(str)
    var.index = var.index.astype(str)

    layers = {}
    for mtx in sorted(export_dir.glob("*.mtx")):
        m = scipy.io.mmread(mtx)               # genes x cells (COO)
        m = sp.csr_matrix(m.T, dtype="float32")  # -> cells x genes, row-sliceable
        assert m.shape == (obs.shape[0], var.shape[0]), (mtx, m.shape)
        layers[mtx.stem] = m

    # Raw counts are the most faithful starting point, so they become X when
    # present; otherwise fall back to the only deposited assay (e.g. Camp).
    primary = "counts" if "counts" in layers else next(iter(layers))
    layer_names = list(layers)  # snapshot: anndata>=0.13 adds a None key (= .X) to the dict
    adata = ad.AnnData(X=layers[primary].copy(), obs=obs, var=var, layers=layers)
    adata.uns["X_source_assay"] = primary
    adata.uns["source_dir"] = str(export_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    print(f"{export_dir} -> {out_path}: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"X = '{primary}', layers = {layer_names}")
    return adata


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
