"""Linear SVM: the project's SVM baseline (proposal V, Table 1).

STATUS: FINAL for the benchmark (decided 2026-11-23: the team leader keeps this
implementation instead of writing a separate one). Model: scikit-learn
``LinearSVC`` (liblinear, one-vs-rest) on the 2,000 scaled HVGs; probability
calibration by **temperature scaling** fitted on donor-grouped out-of-fold
decision scores of the training cells.

Plain-words picture: a Turkish phone charger works fine, but the UK wall socket
has a different shape. You don't rebuild the phone; you buy a small adapter.

Proposal V: "SVM probability calibration is fitted using reference development
data." Table 1: confidence = "Decision scores; reference-calibrated
probabilities"; policy = "Refit training and calibration stages".

THE TRAP: two different "calibrations".
* Probability calibration (temperature scaling) turns decision scores into
  probabilities. It is fitted on *out-of-fold* decision scores from donor-grouped
  inner folds of ``train`` (io_spec §2 "Score transformation - inner folds of train").
* The ``calibration`` role is a separate pile of cells used later for rejection
  thresholds and conformal quantiles, after the score rule is frozen.
This adapter never sees calibration-role cells while fitting: the runner only
hands it training cells, and ``self.calibration_fit_ids`` records exactly which
cells the calibration step used, so a test can prove it.

Why temperature scaling and not one-vs-rest Platt (the Week 2 draft). Platt fits
one sigmoid per class and then divides by their sum. The division deflates the
top probability and can change the predicted class. Temperature scaling learns
one number T and returns softmax(decision / T): the predicted class is exactly
the SVM's, and on leave-one-donor-out folds of the Baron training donors it had
lower log-loss (0.65 vs 1.00), lower ECE (0.13 vs 0.20) and lower Brier score
(0.24 vs 0.32) than Platt (docs/week2_decisions.md, Week 2 notebook §8). Proposal
H2 also names temperature scaling as the supporting calibration analysis.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..provenance import write_json, write_tsv
from .base import Method, MethodOutput, TrainingData, donor_folds


def _decision_matrix(model, rep: np.ndarray) -> np.ndarray:
    """Decision scores as n_cells x n_classes, also for the two-class case."""
    scores = model.decision_function(rep)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LinearSvm(Method):
    name = "linear_svm"

    CALIBRATIONS = ("temperature_inner_folds", "platt_inner_folds", "none")

    def __init__(self, C: float = 1.0, class_weight: str | None = None, representation: str = "scaled_hvg",
                 probability_calibration: str = "temperature_inner_folds", max_iter: int = 5000, dual: bool = True):
        if representation not in {"scaled_hvg", "pca"}:
            raise ValueError("representation must be 'scaled_hvg' or 'pca'")
        if probability_calibration not in self.CALIBRATIONS:
            raise ValueError(f"probability_calibration must be one of {self.CALIBRATIONS}")
        self.C, self.class_weight, self.max_iter, self.dual = float(C), class_weight, int(max_iter), bool(dual)
        self.representation = representation
        self.probability_calibration = probability_calibration
        self.score_type = "calibrated_probability" if probability_calibration != "none" else "decision_score"
        self.calibration_fit_ids: np.ndarray | None = None
        self.temperature: float | None = None
        self.calibrators = None
        self.notes: list[str] = []

    def settings(self) -> dict:
        return {"model": "sklearn LinearSVC (liblinear, one-vs-rest)", "C": self.C,
                "class_weight": self.class_weight, "representation": self.representation,
                "probability_calibration": self.probability_calibration, "max_iter": self.max_iter, "dual": self.dual,
                "temperature": self.temperature, "notes": self.notes}

    # ------------------------------------------------------------------- model
    def _new_classifier(self, seed: int):
        """The classifier object (must offer fit / decision_function / classes_)."""
        from sklearn.svm import LinearSVC

        # dual=True on purpose: on ~3,000 cells x 2,000 dense scaled genes liblinear's primal solver took
        # 75 s to 5+ min per fit, the dual solver 9-22 s (measured on the E1 training block, C=0.01-1).
        return LinearSVC(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter, dual=self.dual,
                         random_state=seed)

    # ------------------------------------------------------------------------- fit
    def fit(self, train: TrainingData, seed: int) -> None:
        labels = np.asarray(train.labels, dtype=str)
        self.vocabulary = sorted(set(labels))
        self.model = self._new_classifier(seed).fit(train.rep, labels)
        if list(self.model.classes_) != self.vocabulary:
            raise ValueError("classifier classes differ from the sorted vocabulary")
        self.calibrators, self.temperature = None, None
        if self.probability_calibration == "none":
            return
        oof = self._out_of_fold_scores(train, labels, seed)
        if oof is None:
            return
        if self.probability_calibration == "temperature_inner_folds":
            self._fit_temperature(oof, labels)
        else:
            self._fit_platt(oof, labels)
        self.calibration_fit_ids = np.asarray(train.cell_ids)[~np.isnan(oof).all(axis=1)]

    # ------------------------------------------------------------- calibration
    def _out_of_fold_scores(self, train: TrainingData, labels: np.ndarray, seed: int) -> np.ndarray | None:
        """Decision scores of every training cell from an SVM that never saw its donor.

        For each held-out training donor: refit the preprocessor (when configured)
        and the SVM on the other training donors, then score the held-out donor.
        These out-of-fold scores are the only data the calibration step sees.
        A class missing from a fold's training cells has no score there (NaN).
        """
        folds = donor_folds(train.groups)
        if not folds:
            self.notes.append("fewer than two training donors: probability calibration skipped")
            self.score_type = "decision_score"
            return None
        oof = np.full((len(labels), len(self.vocabulary)), np.nan)
        for _, fit_idx, out_idx in folds:
            rep_fit, rep_out = train.inner_representation(fit_idx, out_idx)
            inner = self._new_classifier(seed).fit(rep_fit, labels[fit_idx])
            columns = [self.vocabulary.index(c) for c in inner.classes_]
            oof[np.ix_(out_idx, columns)] = _decision_matrix(inner, rep_out)
        return oof

    def _fit_temperature(self, oof: np.ndarray, labels: np.ndarray) -> None:
        """One temperature T minimising the log-loss of softmax(score / T) on out-of-fold scores.

        A class absent from a fold's training cells got no out-of-fold score; it is
        filled with the lowest observed score minus 1, i.e. "that SVM could not vote
        for it". T is searched on a log scale between e^-5 and e^5.
        """
        from scipy.optimize import minimize_scalar

        filled = np.where(np.isnan(oof), np.nanmin(oof) - 1.0, oof)
        target = np.searchsorted(self.vocabulary, labels)
        rows = np.arange(len(labels))

        def log_loss(log_t: float) -> float:
            return float(-np.mean(np.log(_softmax(filled / np.exp(log_t))[rows, target] + 1e-15)))

        result = minimize_scalar(log_loss, bounds=(-5.0, 5.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        if abs(result.x) > 4.9:
            self.notes.append(f"temperature {self.temperature:.4g} is at the edge of the search range")

    def _fit_platt(self, oof: np.ndarray, labels: np.ndarray) -> None:
        """One-vs-rest Platt scaling on out-of-fold scores (kept as a documented alternative)."""
        from sklearn.linear_model import LogisticRegression

        self.calibrators = []
        for j, cls in enumerate(self.vocabulary):
            usable = ~np.isnan(oof[:, j])
            target = labels[usable] == cls
            if target.all() or not target.any():
                # The class never had out-of-fold scores on both sides (e.g. it lives in one donor).
                self.calibrators.append((1.0, 0.0))
                self.notes.append(f"{cls}: no out-of-fold positives and negatives; identity sigmoid used")
                continue
            fit = LogisticRegression(C=1e6, max_iter=1000).fit(oof[usable, j:j + 1], target)
            self.calibrators.append((float(fit.coef_[0, 0]), float(fit.intercept_[0])))

    # --------------------------------------------------------------------- predict
    def predict(self, rep: np.ndarray) -> MethodOutput:
        decision = _decision_matrix(self.model, rep)
        if self.temperature is not None:
            scores = _softmax(decision / self.temperature)
        elif self.calibrators is None:
            scores = decision
        else:
            slope = np.array([a for a, _ in self.calibrators])
            intercept = np.array([b for _, b in self.calibrators])
            with np.errstate(over="ignore"):
                probability = 1.0 / (1.0 + np.exp(-(decision * slope + intercept)))
            total = probability.sum(axis=1, keepdims=True)
            scores = np.where(total > 0, probability / np.where(total > 0, total, 1), 1.0 / len(self.vocabulary))
        labels = np.asarray(self.vocabulary, dtype=object)[np.argmax(scores, axis=1)].astype(str)
        return MethodOutput(self.vocabulary, labels, np.zeros(len(labels), dtype=bool), scores, self.score_type)

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        coef = pd.DataFrame(self.model.coef_.T if self.model.coef_.shape[0] == len(self.vocabulary)
                            else np.column_stack([-self.model.coef_[0], self.model.coef_[0]]),
                            columns=self.vocabulary)
        coef.insert(0, "feature", [f"{self.representation}_{i}" for i in range(coef.shape[0])])
        write_tsv(coef, folder / "svm_coefficients.tsv.gz")
        write_json({"intercept": dict(zip(self.vocabulary, np.atleast_1d(self.model.intercept_).tolist())),
                    "probability_calibration": self.probability_calibration, "temperature": self.temperature,
                    "platt": None if self.calibrators is None else dict(zip(self.vocabulary, self.calibrators)),
                    "calibration_fit_n_cells": None if self.calibration_fit_ids is None else len(self.calibration_fit_ids)},
                   folder / "score_transform.json")
