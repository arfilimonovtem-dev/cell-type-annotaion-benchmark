"""Write the human-readable and machine-readable Week 1 reports."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .integrity import Check, feature_issue_table, results_as_records


def _write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def md_table(frame: pd.DataFrame) -> str:
    def clean(value) -> str:
        return "" if pd.isna(value) else str(value).replace("|", "\\|")

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join(" --- " for _ in frame.columns) + "|"
    body = ["| " + " | ".join(clean(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def write_dataset_report(summary: dict, directory: Path) -> Path:
    name = summary["dataset"]
    _write_json(directory / f"{name}_dataset.json", summary)
    layers = pd.DataFrame(
        [
            {
                "matrix": f"X (copy of {summary['X']['copied_from_layer']})" if key == "X" else f"layer {key}",
                "unit": value["unit"],
                "shape (cells x features)": " x ".join(map(str, value["shape_cells_by_features"])),
                "stored values": value["stored_values"],
                "fraction stored": round(value["fraction_stored"], 4),
                "min": value["min_stored"],
                "max": value["max_stored"],
                "all integer": value["all_integer"],
            }
            for key, value in [("X", summary["X"]), *summary["layers"].items()]
        ]
    )
    lines = [
        f"# Dataset report: {name}",
        "",
        f"- Input: `{summary['input']['relative_path']}` ({summary['input']['bytes']:,} bytes)",
        f"- SHA-256: `{summary['input']['sha256']}`",
        f"- Cells: {summary['n_cells']:,}. Features: {summary['n_features']:,}.",
        f"- Example cell IDs: {', '.join(summary['example_cell_ids'])}",
        f"- Example feature IDs: {', '.join(summary['example_feature_ids'])}",
        "",
        "## Matrices",
        "",
        md_table(layers),
        "",
        "## Cells per donor",
        "",
        md_table(pd.DataFrame(summary["cells_per_donor"].items(), columns=["donor", "cells"])),
        "",
        "## How feature identifiers resolved against NCBI gene_info",
        "",
        md_table(pd.DataFrame(summary["feature_id_resolution"].items(), columns=["resolution", "features"])),
        "",
        "## Cell metadata columns (obs)",
        "",
        md_table(pd.DataFrame(summary["obs_columns"])),
        "",
        "## Feature metadata columns (var)",
        "",
        md_table(pd.DataFrame(summary["var_columns"])),
        "",
    ]
    path = directory / f"{name}_dataset.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_integrity_report(dataset: str, results: list[Check], adata, directory: Path) -> Path:
    _write_json(directory / f"{dataset}_integrity.json", results_as_records(results))
    feature_issue_table(adata).to_csv(directory / f"{dataset}_feature_issues.tsv", sep="\t", index=False)

    def status(check: Check) -> str:
        if check.passed is None:
            return "not run"
        if check.passed:
            return "pass"
        return "FAIL" if check.severity == "fatal" else "review"

    table = pd.DataFrame(
        [
            {
                "group": c.group,
                "check": c.name,
                "severity": c.severity,
                "result": status(c),
                "observed": c.observed,
                "expected": c.expected,
                "detail": c.detail,
            }
            for c in results
        ]
    )
    failed = int((table["result"] == "FAIL").sum())
    review = int((table["result"] == "review").sum())
    lines = [
        f"# Integrity report: {dataset}",
        "",
        f"Checks run: {len(table)}. Fatal failures: {failed}. Warnings to review: {review}.",
        "",
        f"Features flagged for review are listed in `{dataset}_feature_issues.tsv`.",
        "",
        md_table(table),
        "",
    ]
    path = directory / f"{dataset}_integrity.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
