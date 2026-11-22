"""Small invented datasets shaped like the output of load_dataset + attach_labels.

Two datasets ("toyref" and "toyqry") share most genes, so every Week 2 step
(QC, universe, splits, preprocessing, methods, runner) can be tested in a
second without touching real data. Classes are clearly separated by marker
genes, so a correct pipeline reaches 100% accuracy on them.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CLASSES = ["alpha", "beta", "delta"]
N_GENES = 60  # genes 0..59; the first 3*5 are markers (5 per class)


def gene_table(ids: list[int], spike_ins: int = 0) -> pd.DataFrame:
    """A var table as load_dataset would produce it, with Gene IDs '1001', '1002', ..."""
    symbols = [f"G{i}" for i in ids]
    var = pd.DataFrame(
        {
            "feature_id_original": symbols,
            "symbol_original": symbols,
            "symbol": symbols,
            "symbol_repaired": False,
            "ncbi_gene_id": pd.array([str(1000 + i) for i in ids], dtype="string"),
            "id_resolution": "official_symbol",
            "is_spike_in": False,
            "is_mt": False,
            "shares_gene_id": False,
        },
        index=pd.Index(symbols),
    )
    if spike_ins:
        extra = pd.DataFrame(
            {
                "feature_id_original": [f"ERCC-{i:05d}" for i in range(spike_ins)],
                "symbol_original": [f"ERCC-{i:05d}" for i in range(spike_ins)],
                "symbol": [f"ERCC-{i:05d}" for i in range(spike_ins)],
                "symbol_repaired": False,
                "ncbi_gene_id": pd.array([pd.NA] * spike_ins, dtype="string"),
                "id_resolution": "spike_in",
                "is_spike_in": True,
                "is_mt": False,
                "shares_gene_id": False,
            },
            index=pd.Index([f"ERCC-{i:05d}" for i in range(spike_ins)]),
        )
        var = pd.concat([var, extra])
    return var


def make_toy(name: str, donors: list[str], cells_per_donor_class: int = 12, gene_ids: list[int] | None = None,
             seed: int = 0, depth: float = 1.0, classes: list[str] | None = None, spike_ins: int = 0,
             nan_x: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    classes = classes or CLASSES
    gene_ids = list(range(N_GENES)) if gene_ids is None else gene_ids
    names, donor_col, labels, rows = [], [], [], []
    for donor in donors:
        for label in classes:
            for _ in range(cells_per_donor_class):
                rate = np.full(len(gene_ids), 2.0)
                k = CLASSES.index(label) if label in CLASSES else 3
                for j, g in enumerate(gene_ids):
                    if 5 * k <= g < 5 * k + 5:
                        rate[j] = 40.0
                rows.append(rng.poisson(rate * depth))
                names.append(f"{donor}_c{len(names):04d}")
                donor_col.append(donor)
                labels.append(label)
    matrix = np.array(rows, dtype=np.float32)
    if spike_ins:
        matrix = np.hstack([matrix, rng.poisson(3.0, size=(matrix.shape[0], spike_ins)).astype(np.float32)])
    X = sp.csr_matrix(matrix)
    obs = pd.DataFrame(
        {
            "dataset": name,
            "donor": pd.array(donor_col, dtype="string"),
            "label_original": pd.array(labels, dtype="string"),
            "label_harmonized": pd.array(labels, dtype="string"),
            "eligible_primary": True,
            "exclusion_reason": pd.array([pd.NA] * len(names), dtype="string"),
        },
        index=pd.Index(names),
    )
    adata = ad.AnnData(X=X, obs=obs, var=gene_table(gene_ids, spike_ins), layers={"counts": X.copy()})
    if nan_x:
        adata.X = sp.csr_matrix(np.full(adata.shape, np.nan, dtype=np.float32))
    adata.uns["annotbench"] = {"dataset": name, "X_layer": "counts", "assay_unit": "umi_counts",
                               "layer_units": {"counts": "umi_counts"}, "input_relative_path": f"{name}.h5ad"}
    return adata


def with_qc_columns(adata: ad.AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()
    obs["qc_pass"] = True
    obs["qc_reason"] = ""
    return obs


def toy_pair():
    """Reference with 4 donors; query with 2 donors, 2 genes fewer, deeper sequencing."""
    ref = make_toy("toyref", ["r1", "r2", "r3", "r4"], seed=1)
    qry = make_toy("toyqry", ["q1", "q2"], gene_ids=list(range(N_GENES - 2)), seed=2, depth=2.0)
    return ref, qry
