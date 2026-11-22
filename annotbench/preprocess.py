"""Preprocessing: learn the ruler on training cells, glue it shut, apply it to everyone.

Two kinds of steps (docs/io_spec.md §2):

* Stateless, computed per cell from its own numbers: library-size normalization
  and log1p. Applying them to query cells leaks nothing.
* Fitted on ``train`` only, then frozen: highly variable genes (HVG), z-score
  scaling, PCA. Calibration and query cells are only *transformed*.

Proposal IV.E: "Highly variable genes, scaling, PCA, and classifier parameters are
fitted on reference training cells". IV.D: "Original measurements are preserved,
and each method receives its required representation."

Everything here works on new arrays. Nothing edits the caller's matrix in place
(scanpy's ``normalize_total(adata)`` does, which is how a loop over scenarios can
silently feed already-normalized data to the second scenario).

Decisions left to configuration (``preprocess:`` in configs/pilot.yaml):
``target_sum`` (a fixed constant, or ``dataset_median`` as the leader prefers),
which genes define the library size, the HVG method and batch key, the zero-sd
rule, clipping, and the PCA solver.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .provenance import (
    git_state, package_versions, read_tsv, sha256_ids, sha256_ordered, utc_now, write_json, write_tsv,
)

REPRESENTATIONS = ("scaled_hvg", "pca")


class PreprocessError(ValueError):
    """The preprocessor cannot run on these inputs (wrong genes, empty cells, too few cells...)."""


@dataclass(frozen=True)
class PreprocessConfig:
    recipe: str = "lognorm_v1"
    # A number (e.g. 10000) used for every dataset, or "dataset_median": each
    # dataset is scaled to the median library size of its own QC-passing cells.
    target_sum: float | str = 10000
    # "universe": library size = sum over universe genes (identical gene set in
    # every dataset). "all_endogenous": sum over every non-spike-in feature the
    # dataset measured (computed before alignment and passed in).
    library_size: str = "universe"
    n_hvg: int = 2000
    hvg_flavor: str = "seurat"         # "seurat" | "cell_ranger" | "seurat_v3" | "variance"
    hvg_batch_key: str | None = None   # e.g. "donor": choose genes variable within donors
    zero_sd_rule: str = "unit"         # "unit": sd -> 1 for genes constant in train; "drop": remove them
    clip_max: float | None = 10.0      # upper clip after scaling, as scanpy's max_value
    n_pcs: int = 50
    pca_solver: str = "full"

    @classmethod
    def from_config(cls, cfg: dict) -> "PreprocessConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise PreprocessError(f"preprocess: unknown keys {unknown}")
        config = cls(**cfg)
        if not (isinstance(config.target_sum, (int, float)) or config.target_sum == "dataset_median"):
            raise PreprocessError("target_sum must be a number or 'dataset_median'")
        if config.library_size not in {"universe", "all_endogenous"}:
            raise PreprocessError("library_size must be 'universe' or 'all_endogenous'")
        if config.zero_sd_rule not in {"unit", "drop"}:
            raise PreprocessError("zero_sd_rule must be 'unit' or 'drop'")
        return config


def resolve_target_sums(config: PreprocessConfig, pool_totals: dict[str, np.ndarray]) -> dict[str, float]:
    """The normalization constant of each dataset.

    ``pool_totals`` holds the library sizes of each dataset's *normalization
    pool*: its QC-passing cells, whatever their role. The pool does not depend
    on the scenario, so the constant stays fixed across class removals and seeds
    (proposal VI.C: "normalization constants ... remain fixed").

    With ``dataset_median`` the query's constant is computed from unlabeled query
    cells. That is a (mild) query-adaptive step; the run record says so.
    """
    if config.target_sum == "dataset_median":
        return {name: float(np.median(totals)) for name, totals in sorted(pool_totals.items())}
    return {name: float(config.target_sum) for name in sorted(pool_totals)}


def library_sizes(counts) -> np.ndarray:
    return np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()


def lognorm(counts, target_sum: float, totals: np.ndarray | None = None) -> sp.csr_matrix:
    """log1p(count / library size * target_sum), as a new sparse matrix. Pure function."""
    counts = sp.csr_matrix(counts, dtype=np.float64, copy=True)
    totals = library_sizes(counts) if totals is None else np.asarray(totals, dtype=np.float64)
    if totals.shape[0] != counts.shape[0]:
        raise PreprocessError("one library size per cell is required")
    if (totals <= 0).any():
        raise PreprocessError(f"{int((totals <= 0).sum())} cells have a library size of 0; QC should remove them")
    scaled = sp.diags(target_sum / totals) @ counts
    scaled = sp.csr_matrix(scaled)
    np.log1p(scaled.data, out=scaled.data)
    return scaled


def select_hvg(lognormed: sp.csr_matrix, gene_ids: list[str], n_hvg: int, flavor: str,
               batches: np.ndarray | None = None, counts=None) -> pd.DataFrame:
    """Rank genes by variability among the given (training) cells; return the top ``n_hvg``.

    The ranking is done here, not left to the library, so ties are broken by
    Gene ID and the list never depends on chance or on row order.
    """
    n_hvg = min(n_hvg, len(gene_ids))
    gene_index = pd.Index(gene_ids, dtype=str)
    if flavor == "variance":
        mean = np.asarray(lognormed.mean(axis=0)).ravel()
        sq = np.asarray(lognormed.multiply(lognormed).mean(axis=0)).ravel()
        score = pd.Series(sq - mean ** 2, index=gene_index)
        nbatches = pd.Series(0, index=gene_index)
    else:
        import anndata as ad
        import scanpy as sc

        if flavor == "seurat_v3":
            if counts is None:
                raise PreprocessError("seurat_v3 ranks raw counts; pass counts")
            matrix = sp.csr_matrix(counts, dtype=np.float64)
        else:
            matrix = lognormed
        temp = ad.AnnData(X=matrix.copy(), var=pd.DataFrame(index=gene_index))
        batch_key = None
        if batches is not None:
            temp.obs["batch"] = pd.Categorical(np.asarray(batches, dtype=str))
            batch_key = "batch"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = sc.pp.highly_variable_genes(temp, flavor=flavor, n_top_genes=n_hvg, batch_key=batch_key,
                                                inplace=False)
        table.index = gene_index
        column = "variances_norm" if flavor == "seurat_v3" else "dispersions_norm"
        score = table[column].astype(float)
        nbatches = table["highly_variable_nbatches"] if "highly_variable_nbatches" in table else pd.Series(0, index=gene_index)
    ranking = pd.DataFrame({
        "ncbi_gene_id": gene_index,
        "nbatches": np.asarray(nbatches, dtype=float),
        "score": score.to_numpy(dtype=float),
        "id_number": [int(g) if g.isdigit() else -1 for g in gene_index],
    })
    ranking["score_sort"] = ranking["score"].fillna(-np.inf)
    ranking = ranking.sort_values(["nbatches", "score_sort", "id_number", "ncbi_gene_id"],
                                  ascending=[False, False, True, True], kind="stable")
    top = ranking.head(n_hvg).reset_index(drop=True)
    top["rank"] = np.arange(1, len(top) + 1)
    return top[["ncbi_gene_id", "rank", "score", "nbatches"]]


class Preprocessor:
    """Fit on training cells, then transform any cells with the frozen state."""

    def __init__(self, config: PreprocessConfig, seed: int):
        self.config = config
        self.seed = int(seed)
        self.fitted = False

    # ----------------------------------------------------------------- fit
    def fit(self, counts, gene_ids: list[str], cell_ids, dataset: str, target_sums: dict[str, float],
            totals: np.ndarray | None = None, batches: np.ndarray | None = None) -> "Preprocessor":
        from sklearn.decomposition import PCA

        cfg = self.config
        self.gene_ids = [str(g) for g in gene_ids]
        if counts.shape[1] != len(self.gene_ids):
            raise PreprocessError("counts columns and gene_ids differ in length")
        self.universe_sha256 = sha256_ordered(self.gene_ids)
        self.target_sums = dict(target_sums)
        self.reference_dataset = dataset
        if cfg.n_pcs > min(counts.shape[0], cfg.n_hvg, len(self.gene_ids)):
            raise PreprocessError(f"n_pcs={cfg.n_pcs} needs at least {cfg.n_pcs} training cells and HVGs")

        lognormed = lognorm(counts, self._target_sum(dataset), self._totals(counts, totals))
        if cfg.hvg_batch_key and batches is None:
            raise PreprocessError(f"hvg_batch_key={cfg.hvg_batch_key!r} needs per-cell batches")
        hvg = select_hvg(lognormed, self.gene_ids, cfg.n_hvg, cfg.hvg_flavor,
                         batches if cfg.hvg_batch_key else None, counts)
        dense = self._columns(lognormed, hvg["ncbi_gene_id"])
        mean, sd = dense.mean(axis=0), dense.std(axis=0)  # ddof=0, as scikit-learn's StandardScaler
        zero = sd < 1e-12
        if zero.any() and cfg.zero_sd_rule == "drop":
            hvg = hvg.loc[~zero].reset_index(drop=True)
            hvg["rank"] = np.arange(1, len(hvg) + 1)
            dense, mean, sd, zero = dense[:, ~zero], mean[~zero], sd[~zero], zero[~zero]
        sd = np.where(zero, 1.0, sd)
        self.hvg = hvg
        self.scaler = pd.DataFrame({"ncbi_gene_id": hvg["ncbi_gene_id"].to_numpy(), "mean": mean, "sd": sd,
                                    "sd_was_zero": zero})
        scaled = self._scale(dense)
        pca = PCA(n_components=cfg.n_pcs, svd_solver=cfg.pca_solver, random_state=self.seed).fit(scaled)
        self.pca_components = pca.components_            # n_pcs x n_hvg
        self.pca_center = pca.mean_                      # ~0 after scaling, kept for exactness
        self.pca_variance_ratio = pca.explained_variance_ratio_
        self.provenance = {
            "recipe": cfg.recipe,
            "roles_used": ["train"],
            "reference_dataset": dataset,
            "n_cells": int(counts.shape[0]),
            "cell_ids_sha256": sha256_ids(cell_ids),
            "universe_sha256": self.universe_sha256,
            "target_sums": self.target_sums,
            "query_adaptive_constants": cfg.target_sum == "dataset_median",
            **git_state(),
            "package_versions": package_versions(),
            "created_utc": utc_now(),
        }
        self.fitted = True
        return self

    # ----------------------------------------------------------- transform
    def transform(self, counts, gene_ids: list[str], dataset: str, totals: np.ndarray | None = None) -> dict:
        """Apply the frozen state. Nothing is learned here."""
        if not self.fitted:
            raise PreprocessError("fit (or load) the preprocessor first")
        if sha256_ordered([str(g) for g in gene_ids]) != self.universe_sha256:
            raise PreprocessError("these cells are aligned to a different gene universe or order")
        lognormed = lognorm(counts, self._target_sum(dataset), self._totals(counts, totals))
        scaled = self._scale(self._columns(lognormed, self.hvg["ncbi_gene_id"], gene_ids))
        pcs = (scaled - self.pca_center) @ self.pca_components.T
        return {"scaled_hvg": scaled, "pca": pcs}

    # ------------------------------------------------------------- helpers
    def _target_sum(self, dataset: str) -> float:
        if dataset not in self.target_sums:
            raise PreprocessError(f"no normalization constant recorded for dataset '{dataset}'")
        return self.target_sums[dataset]

    def _totals(self, counts, totals):
        if self.config.library_size == "all_endogenous":
            if totals is None:
                raise PreprocessError("library_size='all_endogenous' needs per-cell totals computed before alignment")
            return totals
        return None  # lognorm sums the universe columns it is given

    def _columns(self, lognormed, wanted, gene_ids=None) -> np.ndarray:
        position = {g: i for i, g in enumerate(gene_ids if gene_ids is not None else self.gene_ids)}
        return lognormed[:, [position[g] for g in wanted]].toarray()

    def _scale(self, dense: np.ndarray) -> np.ndarray:
        scaled = (dense - self.scaler["mean"].to_numpy()) / self.scaler["sd"].to_numpy()
        if self.config.clip_max is not None:
            scaled = np.minimum(scaled, self.config.clip_max)
        return scaled

    # --------------------------------------------------------- save / load
    def save(self, folder: str | Path, extra_provenance: dict | None = None) -> Path:
        """Write the fitted-state files named in docs/io_spec.md §2."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        write_json({"config": asdict(self.config), "seed": self.seed, "target_sums": self.target_sums,
                    "reference_dataset": self.reference_dataset, "universe_sha256": self.universe_sha256,
                    "n_universe_genes": len(self.gene_ids)}, folder / "recipe.json")
        write_tsv(self.hvg, folder / "hvg.tsv")
        write_tsv(self.scaler, folder / "scaler.tsv")
        loadings = pd.DataFrame(self.pca_components.T, columns=[f"PC{i + 1}" for i in range(self.pca_components.shape[0])])
        loadings.insert(0, "ncbi_gene_id", self.hvg["ncbi_gene_id"].to_numpy())
        write_tsv(loadings, folder / "pca_loadings.tsv.gz")
        write_tsv(pd.DataFrame({"ncbi_gene_id": self.hvg["ncbi_gene_id"].to_numpy(), "center": self.pca_center}),
                  folder / "pca_center.tsv")
        write_tsv(pd.DataFrame({"pc": [f"PC{i + 1}" for i in range(len(self.pca_variance_ratio))],
                                "explained_variance_ratio": self.pca_variance_ratio}), folder / "pca_variance.tsv")
        write_json({**self.provenance, **(extra_provenance or {})}, folder / "provenance.json")
        return folder

    @classmethod
    def load(cls, folder: str | Path, gene_ids: list[str]) -> "Preprocessor":
        """Rebuild a fitted preprocessor; ``gene_ids`` must be the universe it was fitted on."""
        import json

        folder = Path(folder)
        recipe = json.loads((folder / "recipe.json").read_text())
        self = cls(PreprocessConfig(**recipe["config"]), recipe["seed"])
        self.gene_ids = [str(g) for g in gene_ids]
        self.universe_sha256 = sha256_ordered(self.gene_ids)
        if self.universe_sha256 != recipe["universe_sha256"]:
            raise PreprocessError("gene_ids differ from the universe this preprocessor was fitted on")
        self.target_sums = recipe["target_sums"]
        self.reference_dataset = recipe["reference_dataset"]
        hvg = read_tsv(folder / "hvg.tsv")
        self.hvg = hvg.assign(rank=hvg["rank"].astype(int), score=hvg["score"].astype(float),
                              nbatches=hvg["nbatches"].astype(float))
        scaler = read_tsv(folder / "scaler.tsv")
        self.scaler = scaler.assign(mean=scaler["mean"].astype(float), sd=scaler["sd"].astype(float),
                                    sd_was_zero=scaler["sd_was_zero"] == "True")
        loadings = read_tsv(folder / "pca_loadings.tsv.gz").set_index("ncbi_gene_id").astype(float)
        self.pca_components = loadings.loc[self.hvg["ncbi_gene_id"]].to_numpy().T
        self.pca_center = read_tsv(folder / "pca_center.tsv")["center"].astype(float).to_numpy()
        self.pca_variance_ratio = read_tsv(folder / "pca_variance.tsv")["explained_variance_ratio"].astype(float).to_numpy()
        self.provenance = json.loads((folder / "provenance.json").read_text())
        self.fitted = True
        return self
