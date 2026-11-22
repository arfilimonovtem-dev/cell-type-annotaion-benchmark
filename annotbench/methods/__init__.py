"""Method adapters. Add a new method by writing an adapter and registering it here."""
from __future__ import annotations

from .base import IneligibleInput, Method, MethodOutput, MissingScoreOutput, TrainingData
from .linear_svm import LinearSvm
from .pca_knn import PcaKnn

REGISTRY: dict[str, type[Method]] = {"pca_knn": PcaKnn, "linear_svm": LinearSvm}


def make_method(name: str, params: dict | None) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"unknown method '{name}'. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))


__all__ = ["REGISTRY", "make_method", "Method", "MethodOutput", "TrainingData", "IneligibleInput",
           "MissingScoreOutput"]
