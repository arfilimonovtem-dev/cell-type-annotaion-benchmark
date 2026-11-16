"""Load a dataset into one predictable shape, without altering what was measured."""
from __future__ import annotations

import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import dataset_config, load_registry, require_file, resolve_data_root
from .genes import GeneReference, load_gene_reference, resolve_features


class DatasetLoadError(ValueError):
    """The file exists but does not match what the registry declares."""


def layer_names(adata: ad.AnnData) -> list[str]:
    """Return stored layers. anndata 0.13 also lists None, which only means X."""
    return [name for name in adata.layers.keys() if name is not None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gene_reference_from_registry(registry: dict, root: Path) -> GeneReference:
    return load_gene_reference(require_file(root, registry["gene_info"], "NCBI gene_info table"))


def _matrices_equal(left, right) -> bool:
    """Compare stored values with NaN equal to NaN, so a NaN reaches the integrity report."""
    left, right = sp.csr_matrix(left), sp.csr_matrix(right)
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data, equal_nan=True)
    )


def load_dataset(
    name: str,
    data_root: str | Path | None = None,
    registry: dict | None = None,
    gene_reference: GeneReference | None = None,
) -> ad.AnnData:
    """Read one dataset, set X to its intended representation, and annotate identifiers.

    Every deposited layer stays in adata.layers and every source column stays in
    obs and var. Standard columns are added next to them, never over them.
    """
    root = resolve_data_root(data_root)
    registry = registry or load_registry()
    cfg = dataset_config(registry, name)
    path = require_file(root, cfg["h5ad"], f"{name} expression file")
    adata = ad.read_h5ad(path)

    stored = layer_names(adata)
    deposited_x = adata.uns.get("X_source_assay")
    if deposited_x not in stored or not _matrices_equal(adata.X, adata.layers[deposited_x]):
        raise DatasetLoadError(f"{name}: X is not a copy of a stored layer, so replacing it would lose data")
    use_layer = cfg["expression"]["use_layer"]
    if use_layer not in stored:
        raise DatasetLoadError(f"{name}: configured layer '{use_layer}' not found. Stored layers: {stored}")
    undeclared = sorted(set(stored) - set(cfg["layer_units"]))
    if undeclared:
        raise DatasetLoadError(f"{name}: layers with no declared unit in the registry: {undeclared}")
    adata.X = adata.layers[use_layer].copy()

    columns = cfg["columns"]
    features = cfg["features"]
    absent = [c for c in (columns["donor"], columns["label"]) if c not in adata.obs]
    absent += [c for c in (features.get("symbol_column"), features.get("spike_in_column")) if c and c not in adata.var]
    if absent:
        raise DatasetLoadError(
            f"{name}: expected metadata columns are absent: {absent}. "
            f"The file has {adata.n_obs} rows x {adata.n_vars} columns; check whether cells and genes are swapped."
        )
    for standard in ("dataset", "label_original"):
        if standard in adata.obs:
            raise DatasetLoadError(f"{name}: source obs already has a column called '{standard}'")
    donor = columns.get("donor_prefix", "") + adata.obs[columns["donor"]].astype(str)
    if "donor" in adata.obs and not donor.equals(adata.obs["donor"].astype(str)):
        raise DatasetLoadError(f"{name}: source column 'donor' disagrees with the standard donor value")
    adata.obs["dataset"] = name
    adata.obs["donor"] = donor.astype("string")
    adata.obs["label_original"] = adata.obs[columns["label"]].astype("string")

    symbols = adata.var[features["symbol_column"]] if features.get("symbol_column") else adata.var_names.to_series()
    spike_column = features.get("spike_in_column")
    spike_in = adata.var[spike_column].astype(bool) if spike_column else pd.Series(False, index=adata.var_names)
    resolved = resolve_features(
        adata.var_names,
        symbols,
        gene_reference or load_gene_reference_from_registry(registry, root),
        r_mangled_names=features["r_mangled_names"],
        spike_in=spike_in,
    )
    for column in resolved.columns:
        adata.var[column] = resolved[column]

    adata.uns["annotbench"] = {
        "dataset": name,
        "X_layer": use_layer,
        "assay_unit": cfg["layer_units"][use_layer],
        "layer_units": dict(cfg["layer_units"]),
        "input_relative_path": cfg["h5ad"],
    }
    return adata


def _matrix_summary(matrix, unit: str) -> dict:
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    return {
        "unit": unit,
        "shape_cells_by_features": [int(matrix.shape[0]), int(matrix.shape[1])],
        "storage": type(matrix).__name__,
        "value_type": str(matrix.dtype),
        "stored_values": int(values.size),
        "fraction_stored": float(values.size / (matrix.shape[0] * matrix.shape[1])),
        "min_stored": float(values.min()) if values.size else None,
        "max_stored": float(values.max()) if values.size else None,
        "all_integer": bool(np.all(values == np.round(values))),
    }


def _column_summary(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "name": str(column),
            "type": str(frame[column].dtype),
            "missing": int(frame[column].isna().sum()),
            "distinct": int(frame[column].nunique(dropna=True)),
        }
        for column in frame.columns
    ]


def describe_dataset(adata: ad.AnnData, data_root: str | Path | None = None) -> dict:
    """Summarize dimensions, layers and metadata for the dataset report."""
    info = adata.uns["annotbench"]
    root = resolve_data_root(data_root)
    path = root / info["input_relative_path"]
    resolution = adata.var["id_resolution"].value_counts()
    return {
        "dataset": info["dataset"],
        "input": {
            "relative_path": info["input_relative_path"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "X": {"copied_from_layer": info["X_layer"], **_matrix_summary(adata.X, info["assay_unit"])},
        "layers": {
            name: _matrix_summary(adata.layers[name], info["layer_units"][name]) for name in layer_names(adata)
        },
        "cells_per_donor": {str(k): int(v) for k, v in adata.obs["donor"].value_counts().sort_index().items()},
        "feature_id_resolution": {str(k): int(v) for k, v in resolution.sort_index().items()},
        "obs_columns": _column_summary(adata.obs),
        "var_columns": _column_summary(adata.var),
        "example_cell_ids": [str(x) for x in adata.obs_names[:3]],
        "example_feature_ids": [str(x) for x in adata.var_names[:3]],
    }
