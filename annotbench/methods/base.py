"""The common socket every annotation method plugs into.

Proposal IX: "Adapters expose a common schema for cell identifiers, class
vocabularies, predictions, scores, rejection status, runtime, and errors."

The runner only ever talks to ``Method``. It never knows whether the inside is
kNN, an SVM or, later, SingleR. Division of labour:

* The runner owns the protocol: which cells, which genes, the frozen
  preprocessor, timing, failure records and output files. That is what makes
  the comparison fair: every method gets the same inputs.
* The adapter owns the method: it declares which representation it eats
  (``pca``, ``scaled_hvg`` or raw ``counts``) and which input units it accepts,
  fits on training cells, and returns scores in sorted-vocabulary order.

An adapter never receives the labels of calibration or query cells.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

PROBABILITY_SCORES = {"probability", "calibrated_probability"}
SCORE_TYPES = {"probability", "calibrated_probability", "decision_score", "vote_fraction", "similarity",
               "correlation", "prediction_score"}  # docs/io_spec.md §4


class IneligibleInput(Exception):
    """The inputs violate the method's declared requirements (record status 'ineligible_input')."""


class MissingScoreOutput(Exception):
    """The method produced labels but no defensible per-class scores (status 'missing_score_output')."""


@dataclass
class TrainingData:
    """What a method may learn from: training cells only.

    ``inner_representation(fit_idx, apply_idx)`` returns the method's representation
    for two subsets of the training cells, with the preprocessor *refitted on
    fit_idx only* when the config asks for strict inner cross-validation. Use it
    for anything tuned or calibrated inside ``train`` (k for kNN, Platt scaling
    for the SVM), so held-out inner cells never shaped the HVGs, scaler or PCA.
    """

    rep: np.ndarray
    labels: np.ndarray
    cell_ids: np.ndarray
    groups: np.ndarray  # inner_fold from the split file (the donor)
    inner_representation: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class MethodOutput:
    vocabulary: list[str]
    predicted_label: np.ndarray          # "" where rejected
    rejected: np.ndarray                 # bool
    scores: np.ndarray | None            # n_cells x len(vocabulary), or None
    score_type: str | None


class Method(ABC):
    name: str = "abstract"
    representation: str = "pca"                 # "pca" | "scaled_hvg" | "counts"
    accepted_units: frozenset = frozenset({"umi_counts"})
    score_type: str | None = None

    @abstractmethod
    def fit(self, train: TrainingData, seed: int) -> None: ...

    @abstractmethod
    def predict(self, rep: np.ndarray) -> MethodOutput: ...

    @abstractmethod
    def settings(self) -> dict:
        """Every setting that changes the output ("Record settings")."""

    def save(self, folder: Path) -> None:
        """Write the method's native fitted state (optional)."""

    def check_units(self, units: dict[str, str]) -> None:
        bad = {d: u for d, u in units.items() if u not in self.accepted_units}
        if bad:
            raise IneligibleInput(f"{self.name} accepts {sorted(self.accepted_units)}, got {bad}")


def donor_folds(groups: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-group-out folds over training cells, in sorted group order.

    With fewer than two groups there is nothing to hold out, so an empty list is
    returned and the caller must fall back (and record that it did).
    """
    groups = np.asarray(groups, dtype=str)
    names = sorted(set(groups))
    if len(names) < 2:
        return []
    return [(g, np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in names]


def check_output(output: MethodOutput, n_cells: int) -> None:
    """The contract every adapter's output must meet before the runner writes it."""
    vocab = output.vocabulary
    if vocab != sorted(vocab) or len(set(vocab)) != len(vocab):
        raise ValueError("vocabulary must be sorted and unique")
    if len(output.predicted_label) != n_cells or len(output.rejected) != n_cells:
        raise ValueError("one prediction per cell is required")
    allowed = set(vocab) | {""}
    bad = set(map(str, output.predicted_label)) - allowed
    if bad:
        raise ValueError(f"predicted labels outside the vocabulary: {sorted(bad)[:5]}")
    if np.any((np.asarray(output.predicted_label) == "") != np.asarray(output.rejected, dtype=bool)):
        raise ValueError("a rejected cell must have an empty label, and only rejected cells may")
    if output.scores is None:
        return
    if output.score_type not in SCORE_TYPES:
        raise ValueError(f"unknown score_type {output.score_type!r}")
    if output.scores.shape != (n_cells, len(vocab)):
        raise ValueError(f"scores shape {output.scores.shape}, expected {(n_cells, len(vocab))}")
    if not np.all(np.isfinite(output.scores)):
        raise ValueError("scores contain NaN or infinity")
    if output.score_type in PROBABILITY_SCORES | {"vote_fraction"}:
        if not np.allclose(output.scores.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError(f"{output.score_type} rows must sum to 1")
