"""Check that this machine has what the pilot needs, and that its inputs are intact."""
from __future__ import annotations

import platform
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .loading import sha256_file

PYTHON_TESTED = "3.12.2"
PYTHON_PACKAGES = {
    "anndata": "0.13.4",
    "numpy": "2.4.6",
    "pandas": "3.0.6",
    "scipy": "1.16.0",
    "h5py": "3.15.1",
    "PyYAML": "6.0.3",
    "GEOparse": "2.0.4",
    "scanpy": "1.12.4",        # used from the preprocessing step onward
    "scikit-learn": "1.7.1",   # used by the SVM and PCA-kNN baselines
}
R_PACKAGES = {"SingleCellExperiment": "1.24.0", "SummarizedExperiment": "1.32.0", "Matrix": "1.6.5"}


def check_python() -> pd.DataFrame:
    rows = [{"tool": "python", "tested": PYTHON_TESTED, "found": platform.python_version()}]
    for package, tested in PYTHON_PACKAGES.items():
        try:
            found = version(package)
        except PackageNotFoundError:
            found = "MISSING"
        rows.append({"tool": package, "tested": tested, "found": found})
    return pd.DataFrame(rows)


def check_r() -> pd.DataFrame:
    """R is needed only to re-export Baron from its .rds file."""
    if shutil.which("Rscript") is None:
        return pd.DataFrame([{"tool": "Rscript", "tested": "R 4.3.3", "found": "MISSING"}])
    script = 'for (p in commandArgs(TRUE)) cat(p, tryCatch(as.character(packageVersion(p)), error = function(e) "MISSING"), "\\n")'
    done = subprocess.run(
        ["Rscript", "-e", script, "--args", *R_PACKAGES], capture_output=True, text=True, check=True
    )
    found = dict(line.split()[:2] for line in done.stdout.splitlines() if line.strip())
    return pd.DataFrame([{"tool": p, "tested": t, "found": found.get(p, "MISSING")} for p, t in R_PACKAGES.items()])


def check_inputs(root: Path, manifest_path: Path) -> pd.DataFrame:
    """Compare every required input with its recorded size and checksum."""
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    statuses = []
    for row in manifest.itertuples(index=False):
        path = root / row.relative_path
        if not path.is_file():
            statuses.append("MISSING")
        elif str(path.stat().st_size) != row.bytes or sha256_file(path) != row.sha256:
            statuses.append("CHANGED")
        else:
            statuses.append("ok")
    return manifest.assign(status=statuses)[["dataset", "relative_path", "bytes", "status", "source"]]
