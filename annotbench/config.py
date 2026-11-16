"""Locate the data folder and read the project's settings files.

No other module builds an absolute data path. A teammate whose data lives
somewhere else sets one environment variable, or passes --data-root, and the
same code runs unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT_ENV = "ANNOTBENCH_DATA_ROOT"
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "datasets.yaml"
DEFAULT_LABEL_MAP = REPO_ROOT / "labels" / "label_map.csv"
DEFAULT_INPUT_MANIFEST = REPO_ROOT / "manifests" / "pilot_inputs.tsv"
DEFAULT_REPORTS = REPO_ROOT / "reports" / "week1"


class MissingInputError(FileNotFoundError):
    """A file or folder the pipeline needs is not where the settings say."""


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """Return the data folder: --data-root, then the environment variable, then <repo>/data."""
    if explicit is not None:
        root, source = Path(explicit), "--data-root"
    elif os.environ.get(DATA_ROOT_ENV):
        root, source = Path(os.environ[DATA_ROOT_ENV]), DATA_ROOT_ENV
    else:
        root, source = REPO_ROOT / "data", "the default location"
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise MissingInputError(
            f"Data folder not found: {root} (taken from {source}). "
            f"Pass --data-root or set {DATA_ROOT_ENV}. See docs/environment.md."
        )
    return root


def require_file(root: Path, relative: str, what: str) -> Path:
    """Return root/relative, or stop with a message saying what is missing."""
    path = root / relative
    if not path.is_file():
        raise MissingInputError(
            f"Missing {what}: expected '{relative}' under the data folder {root}. "
            "See docs/environment.md for where each input comes from."
        )
    return path


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_registry(path: str | Path | None = None) -> dict:
    return load_yaml(path or DEFAULT_REGISTRY)


def dataset_config(registry: dict, name: str) -> dict:
    datasets = registry["datasets"]
    if name not in datasets:
        raise ValueError(f"Unknown dataset '{name}'. Known datasets: {sorted(datasets)}")
    return datasets[name]
