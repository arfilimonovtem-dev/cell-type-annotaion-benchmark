"""Small helpers that make every Week 2 output traceable and byte-reproducible.

Two rules live here so no other module has to remember them:

1. Anything that identifies a set of cells or genes is a SHA-256 of the *sorted*
   identifiers, so the hash does not depend on row order.
2. Compressed tables are written with a fixed gzip timestamp (mtime=0). A .gz
   header normally stores the time of writing, which would make two identical
   tables written a second apart have different checksums.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import REPO_ROOT

GZIP_FIXED = {"method": "gzip", "mtime": 0}


def FLOAT_FORMAT(value) -> str:
    """Shortest text that reads back as exactly the same float (Python's repr), e.g. 0.9 not 0.90000000000000002."""
    return repr(float(value))
TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "anndata", "scikit-learn", "scanpy")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: Iterable) -> str:
    """Hash of the sorted identifiers: the same set gives the same hash in any order."""
    return sha256_text("\n".join(sorted(str(i) for i in ids)))


def sha256_ordered(ids: Iterable) -> str:
    """Hash of identifiers in the given order (use it when order itself matters, e.g. gene columns)."""
    return sha256_text("\n".join(str(i) for i in ids))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_content(path: str | Path) -> str:
    """Hash of a file's content, decompressed first if it is gzipped."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    digest = hashlib.sha256()
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a TSV deterministically (fixed gzip mtime, full float precision, '\\n' line ends)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = GZIP_FIXED if path.suffix == ".gz" else None
    frame.to_csv(path, sep="\t", index=index, float_format=FLOAT_FORMAT, lineterminator="\n",
                 compression=compression)
    return path


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a TSV with every column as text and no silent NaN conversion of empty strings."""
    kwargs.setdefault("dtype", str)
    kwargs.setdefault("keep_default_na", False)
    return pd.read_csv(path, sep="\t", **kwargs)


def write_json(content, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _json_default(value):
    # numpy scalars and paths are the usual things json cannot serialise by itself
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot write {type(value).__name__} to JSON")


def git_state() -> dict:
    """Current commit and whether the working tree has uncommitted changes."""
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    commit = run("rev-parse", "--short", "HEAD")
    status = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict:
    found = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def host_label() -> str:
    return platform.node() or "unknown"
