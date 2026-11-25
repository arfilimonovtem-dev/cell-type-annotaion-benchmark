"""Method adapters. Add a new method by writing an adapter and registering it here.

A config can run several *instances* of one method, e.g. kNN with k=5 and k=10:

    methods:
      pca_knn.k5:  {method: pca_knn, k: 5}
      pca_knn.k10: {method: pca_knn, k: 10}
      linear_svm:  {C: 1.0}            # no 'method' key: the instance name is the method

The instance name goes into the run ID (``__method-pca_knn.k5``); the method
family is recorded separately in the run record.
"""
from __future__ import annotations

from .base import IneligibleInput, Method, MethodOutput, MissingScoreOutput, TrainingData
from .linear_svm import LinearSvm
from .pca_knn import PcaKnn

REGISTRY: dict[str, type[Method]] = {"pca_knn": PcaKnn, "linear_svm": LinearSvm}


def resolve_instance(instance: str, params: dict | None) -> tuple[str, dict]:
    """(method family, constructor keyword arguments) for one configured instance."""
    kwargs = dict(params or {})
    family = str(kwargs.pop("method", instance))
    return family, kwargs


def make_method(instance: str, params: dict | None) -> Method:
    family, kwargs = resolve_instance(instance, params)
    if family not in REGISTRY:
        raise KeyError(f"unknown method '{family}' (instance '{instance}'). Registered: {sorted(REGISTRY)}")
    method = REGISTRY[family](**kwargs)
    method.instance = instance
    return method


__all__ = ["REGISTRY", "make_method", "resolve_instance", "Method", "MethodOutput", "TrainingData",
           "IneligibleInput", "MissingScoreOutput"]
