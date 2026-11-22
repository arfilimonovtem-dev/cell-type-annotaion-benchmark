"""PCA-kNN: label a cell by a vote of its k nearest training cells in PCA space.

Plain-words picture: a new child joins the playground; ask the 15 children
standing closest which team they are on. 12 say blue, 3 say red -> guess blue,
and write down 12/15 = 0.8 blue, 3/15 = 0.2 red. Those are vote fractions.

Proposal V: "kNN vote fractions are identified as score outputs rather than
assumed calibrated probabilities." -> ``score_type = "vote_fraction"``.
Table 1: perturbation policy "Reference-fitted PCA and neighbor index" -> the
PCA comes from the runner's train-fitted preprocessor, the index from training
cells only.

Determinism: ``algorithm="brute"`` is exact; the runner passes cells sorted by
ID; a tie in votes goes to the class that comes first in the sorted vocabulary
(``np.argmax`` returns the first maximum).

Decisions left to configuration: ``k``, uniform vs distance weights, the metric,
and whether k is chosen by donor-grouped inner cross-validation (``k_grid``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..provenance import write_tsv
from .base import IneligibleInput, Method, MethodOutput, TrainingData, donor_folds


def vote_fractions(neighbor_classes: np.ndarray, n_classes: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Row i: share of the (weighted) votes of cell i's neighbours for each class."""
    n_cells, k = neighbor_classes.shape
    weights = np.ones((n_cells, k)) if weights is None else weights
    votes = np.zeros((n_cells, n_classes))
    rows = np.repeat(np.arange(n_cells), k)
    np.add.at(votes, (rows, neighbor_classes.ravel()), weights.ravel())
    return votes / votes.sum(axis=1, keepdims=True)


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Macro-F1 over the classes present in ``truth`` (a class never predicted scores 0)."""
    from sklearn.metrics import f1_score

    classes = sorted(set(truth))
    return float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0))


class PcaKnn(Method):
    name = "pca_knn"
    representation = "pca"
    score_type = "vote_fraction"

    def __init__(self, k: int = 15, weights: str = "uniform", metric: str = "euclidean",
                 k_grid: list[int] | None = None):
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k, self.weights, self.metric = int(k), weights, metric
        self.k_grid = sorted(int(x) for x in k_grid) if k_grid else None
        self.tuning: pd.DataFrame | None = None

    def settings(self) -> dict:
        return {"k": self.k, "weights": self.weights, "metric": self.metric, "algorithm": "brute",
                "k_grid": self.k_grid, "k_selected_by": "inner_cv" if self.k_grid else "config",
                "tie_rule": "first class in sorted vocabulary"}

    def fit(self, train: TrainingData, seed: int) -> None:
        from sklearn.neighbors import NearestNeighbors

        if self.k_grid:
            self.k, self.tuning = select_k(train, self.k_grid, self.weights, self.metric)
        if len(train.labels) < self.k:
            raise IneligibleInput(f"k={self.k} but only {len(train.labels)} training cells")
        self.vocabulary = sorted(set(map(str, train.labels)))
        self.train_classes = np.searchsorted(self.vocabulary, np.asarray(train.labels, dtype=str))
        self.train_ids = np.asarray(train.cell_ids)
        self.train_rep = np.asarray(train.rep)
        self.index = NearestNeighbors(n_neighbors=self.k, algorithm="brute", metric=self.metric).fit(self.train_rep)

    def predict(self, rep: np.ndarray) -> MethodOutput:
        distance, neighbors = self.index.kneighbors(rep, n_neighbors=self.k)
        weights = 1.0 / np.maximum(distance, 1e-12) if self.weights == "distance" else None
        scores = vote_fractions(self.train_classes[neighbors], len(self.vocabulary), weights)
        labels = np.asarray(self.vocabulary, dtype=object)[np.argmax(scores, axis=1)].astype(str)
        return MethodOutput(self.vocabulary, labels, np.zeros(len(labels), dtype=bool), scores, self.score_type)

    def save(self, folder: Path) -> None:
        table = pd.DataFrame(self.train_rep, columns=[f"PC{i + 1}" for i in range(self.train_rep.shape[1])])
        table.insert(0, "label", np.asarray(self.vocabulary)[self.train_classes])
        table.insert(0, "cell_id", self.train_ids)
        write_tsv(table, Path(folder) / "knn_reference.tsv.gz")
        if self.tuning is not None:
            write_tsv(self.tuning, Path(folder) / "k_tuning.tsv")


def select_k(train: TrainingData, k_grid: list[int], weights: str = "uniform", metric: str = "euclidean"):
    """Choose k by leave-one-donor-out inner CV on training cells (proposal IV.E).

    For each held-out training donor, the preprocessor is refitted on the other
    training donors (via ``train.inner_representation``), every k in the grid is
    scored by macro-F1 on the held-out donor, and the k with the best mean wins.
    Ties go to the smaller k. Returns (k, table of per-fold scores).
    """
    from sklearn.neighbors import NearestNeighbors

    folds = donor_folds(train.groups)
    if not folds:
        raise IneligibleInput("k_grid needs at least two training donors for donor-grouped inner CV")
    labels = np.asarray(train.labels, dtype=str)
    rows = []
    for donor, fit_idx, out_idx in folds:
        rep_fit, rep_out = train.inner_representation(fit_idx, out_idx)
        vocab = sorted(set(labels[fit_idx]))
        classes = np.searchsorted(vocab, labels[fit_idx])
        largest = min(max(k_grid), len(fit_idx))
        distance, neighbors = NearestNeighbors(n_neighbors=largest, algorithm="brute", metric=metric) \
            .fit(rep_fit).kneighbors(rep_out)
        for k in k_grid:
            if k > largest:
                continue
            w = 1.0 / np.maximum(distance[:, :k], 1e-12) if weights == "distance" else None
            votes = vote_fractions(classes[neighbors[:, :k]], len(vocab), w)
            predicted = np.asarray(vocab)[np.argmax(votes, axis=1)]
            rows.append({"held_out_donor": donor, "k": k, "n_held_out": len(out_idx),
                         "macro_f1": macro_f1(labels[out_idx], predicted),
                         "accuracy": float(np.mean(predicted == labels[out_idx]))})
    table = pd.DataFrame(rows)
    summary = table.groupby("k")["macro_f1"].mean()
    best = int(summary[summary == summary.max()].index.min())
    return best, table
