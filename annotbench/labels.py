"""Read the versioned label map and attach harmonized labels to cells."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_LABEL_MAP

REQUIRED_COLUMNS = [
    "dataset", "original_label", "harmonized_label", "cl_id", "cl_id_verified",
    "evidence", "ambiguity", "eligible_primary", "exclusion_reason",
]
EXCLUSION_REASONS = {"author_qc_fail", "uncertain_label", "suspected_doublet", "unresolved_mapping"}


class LabelMapError(ValueError):
    """The label map is malformed, or the data holds a label it does not list."""


def load_label_map(path: str | Path | None = None) -> pd.DataFrame:
    table = pd.read_csv(path or DEFAULT_LABEL_MAP, dtype=str, keep_default_na=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise LabelMapError(f"label map lacks columns: {missing}")

    flags = table["eligible_primary"].str.lower()
    if not flags.isin(["true", "false"]).all():
        raise LabelMapError("eligible_primary must be 'true' or 'false' on every row")
    table["eligible_primary"] = flags == "true"

    key = ["dataset", "original_label"]
    if table.duplicated(key).any():
        raise LabelMapError(f"duplicate rows in label map: {table.loc[table.duplicated(key), key].values.tolist()}")

    eligible = table["eligible_primary"]
    if (table.loc[eligible, "harmonized_label"] == "").any():
        raise LabelMapError("every eligible label needs a harmonized_label")
    if (table.loc[eligible, "exclusion_reason"] != "").any():
        raise LabelMapError("an eligible label cannot have an exclusion_reason")
    unknown = set(table.loc[~eligible, "exclusion_reason"]) - EXCLUSION_REASONS
    if unknown:
        raise LabelMapError(f"unknown or empty exclusion_reason values: {sorted(unknown)}")
    return table


def unknown_labels(obs: pd.DataFrame, dataset: str, label_map: pd.DataFrame) -> list[str]:
    """Return labels present in the data but absent from the map."""
    listed = set(label_map.loc[label_map["dataset"] == dataset, "original_label"])
    return sorted(set(obs["label_original"].dropna()) - listed)


def harmonize_obs(
    obs: pd.DataFrame, dataset: str, label_map: pd.DataFrame, missing_label_reason: str | None
) -> pd.DataFrame:
    """Return harmonized_label, eligible_primary and exclusion_reason for each cell.

    A dictionary lookup would turn an unlisted label into a silent NaN, so an
    unlisted label stops the run here instead.
    """
    unknown = unknown_labels(obs, dataset, label_map)
    if unknown:
        raise LabelMapError(f"{dataset}: labels missing from label map: {unknown}")
    original = obs["label_original"]
    if original.isna().any() and missing_label_reason is None:
        raise LabelMapError(f"{dataset}: {int(original.isna().sum())} cells have no label and no declared reason")

    rows = label_map[label_map["dataset"] == dataset].set_index("original_label")
    harmonized = original.map(rows["harmonized_label"]).astype("string").replace("", pd.NA)
    eligible = original.map(rows["eligible_primary"].to_dict()).astype("boolean").fillna(False).astype(bool)
    reason = original.map(rows["exclusion_reason"]).astype("string").replace("", pd.NA)
    reason = reason.mask(original.isna(), missing_label_reason)
    return pd.DataFrame(
        {"label_harmonized": harmonized, "eligible_primary": eligible, "exclusion_reason": reason},
        index=obs.index,
    )


def attach_labels(adata, label_map: pd.DataFrame, cfg: dict) -> None:
    """Add the harmonized label columns to adata.obs. The original label stays untouched."""
    added = harmonize_obs(
        adata.obs, adata.uns["annotbench"]["dataset"], label_map, cfg["columns"].get("missing_label_means")
    )
    for column in added.columns:
        adata.obs[column] = added[column]
