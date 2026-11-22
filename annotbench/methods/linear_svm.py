"""Linear SVM adapter: the socket for the team leader's SVM.

STATUS: ADAPTER + PLACEHOLDER MODEL. The leader is writing the real SVM. The
adapter part (class attributes, ``settings``, the output contract, the rule
about which cells the probability calibration may see) is meant to stay. The
two marked ``LEADER:`` blocks are where the leader's code goes. Until then a
plain scikit-learn ``LinearSVC`` stands in, so the runner can be tested end to end.

Plain-words picture: a Turkish phone charger works fine, but the UK wall socket
has a different shape. You don't rebuild the phone; you buy a small adapter.

Proposal V: "SVM probability calibration is fitted using reference development
data." Table 1: confidence = "Decision scores; reference-calibrated
probabilities"; policy = "Refit training and calibration stages".

THE TRAP: two different "calibrations".
* Probability calibration (Platt scaling) turns decision scores into
  probabilities. It is fitted on *out-of-fold* decision scores from donor-grouped
  inner folds of ``train`` (io_spec §2 "Score transformation - inner folds of train").
* The ``calibration`` role is a separate pile of cells used later for rejection
  thresholds and conformal quantiles, after the score rule is frozen.
This adapter never sees calibration-role cells while fitting: the runner only
hands it training cells, and ``self.calibration_fit_ids`` records exactly which
cells the Platt step used, so a test can prove it.

Open questions for the leader (docs/week2_decisions.md): representation
(``scaled_hvg`` or ``pca``), C and class weights fixed or tuned, and whether the
calibration is one-vs-rest Platt (as here), multinomial, or temperature scaling.
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


class LinearSvm(Method):
    name = "linear_svm"

    def __init__(self, C: float = 1.0, class_weight: str | None = None, representation: str = "scaled_hvg",
                 probability_calibration: str = "platt_inner_folds", max_iter: int = 5000, dual: bool = True):
        if representation not in {"scaled_hvg", "pca"}:
            raise ValueError("representation must be 'scaled_hvg' or 'pca'")
        if probability_calibration not in {"platt_inner_folds", "none"}:
            raise ValueError("probability_calibration must be 'platt_inner_folds' or 'none'")
        self.C, self.class_weight, self.max_iter, self.dual = float(C), class_weight, int(max_iter), bool(dual)
        self.representation = representation
        self.probability_calibration = probability_calibration
        self.score_type = "calibrated_probability" if probability_calibration != "none" else "decision_score"
        self.calibration_fit_ids: np.ndarray | None = None
        self.notes: list[str] = []

    def settings(self) -> dict:
        return {"model": "PLACEHOLDER sklearn LinearSVC (replace with the leader's SVM)", "C": self.C,
                "class_weight": self.class_weight, "representation": self.representation,
                "probability_calibration": self.probability_calibration, "max_iter": self.max_iter, "dual": self.dual,
                "notes": self.notes}

    # ---------------------------------------------------------------- LEADER: model
    def _new_classifier(self, seed: int):
        """LEADER: return your classifier object here (must offer fit / decision_function / classes_)."""
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
        self.calibrators = None
        if self.probability_calibration == "platt_inner_folds":
            self._fit_platt(train, labels, seed)

    # ---------------------------------------------------------------- LEADER: calibration
    def _fit_platt(self, train: TrainingData, labels: np.ndarray, seed: int) -> None:
        """One-vs-rest Platt scaling on out-of-fold decision scores (donor-grouped inner folds).

        For each held-out training donor: refit the preprocessor and the SVM on the
        other training donors, score the held-out donor. The resulting out-of-fold
        scores are the only data the sigmoids see. LEADER: replace if your SVM
        calibrates differently, but keep "inner folds of train only".
        """
        from sklearn.linear_model import LogisticRegression

        folds = donor_folds(train.groups)
        if not folds:
            self.notes.append("fewer than two training donors: probability calibration skipped")
            self.score_type = "decision_score"
            return
        oof = np.full((len(labels), len(self.vocabulary)), np.nan)
        for _, fit_idx, out_idx in folds:
            rep_fit, rep_out = train.inner_representation(fit_idx, out_idx)
            inner = self._new_classifier(seed).fit(rep_fit, labels[fit_idx])
            columns = [self.vocabulary.index(c) for c in inner.classes_]
            oof[np.ix_(out_idx, columns)] = _decision_matrix(inner, rep_out)
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
        self.calibration_fit_ids = np.asarray(train.cell_ids)[~np.isnan(oof).all(axis=1)]

    # --------------------------------------------------------------------- predict
    def predict(self, rep: np.ndarray) -> MethodOutput:
        decision = _decision_matrix(self.model, rep)
        if self.calibrators is None:
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
                    "platt": None if self.calibrators is None else dict(zip(self.vocabulary, self.calibrators)),
                    "calibration_fit_n_cells": None if self.calibration_fit_ids is None else len(self.calibration_fit_ids)},
                   folder / "score_transform.json")
