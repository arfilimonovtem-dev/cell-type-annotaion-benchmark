"""Per-cell quality control: compute metrics, apply prespecified rules, report retention.

Why QC sits here, before the split step
---------------------------------------
QC decides which cells are usable at all. A cell that fails QC becomes role
``excluded`` in every scenario, so QC must run before splits, and its verdict
must not depend on the scenario (otherwise "the same cell" would be usable in
one experiment and not in another).

Why QC may look at query cells without leaking
----------------------------------------------
Every rule here is a *fixed* threshold applied to one cell's own numbers, so a
cell's verdict depends on nothing but that cell. That is "stateless", like
library-size normalization. A data-driven rule (for example "median +/- 3 MAD of
this dataset") would make each cell's verdict depend on its batch-mates, and on
the query that is a query-adaptive step that must be declared (proposal IV.E).

Proposal IV.D: "QC will use assay-aware prespecified rules, with retention
reported by dataset, donor, and class." and "Mitochondrial filtering is used only
where the feature set supports it."

What is deliberately NOT decided here
-------------------------------------
The threshold values. They come from ``configs/pilot.yaml`` (``qc:``), with a
per-dataset override, because the leader wants to fix them after a per-dataset
EDA. A rule set to ``null`` is skipped, and the report says so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Symbol patterns for gene sets. They live in the config; these are only defaults.
# Careful: a naive '^HB' would also catch HBEGF (a growth factor) and HBP1 (a
# transcription factor), and '^RP[SL]' would catch pseudogenes such as RPL13AP5.
DEFAULT_GENE_SET_PATTERNS = {
    "ribo": r"^(RPL|RPS)\d+[A-Z]?\d*$|^RPLP[012]$|^RPSA$",
    "hb": r"^HB[ABDEGMQZ]\d*$",
}

# metric column -> (rule field, direction, failure label)
RULES = {
    "min_total_counts": ("total_counts", "min", "low_total_counts"),
    "max_total_counts": ("total_counts", "max", "high_total_counts"),
    "min_genes_detected": ("n_genes_detected", "min", "low_genes_detected"),
    "max_genes_detected": ("n_genes_detected", "max", "high_genes_detected"),
    "max_pct_mt": ("pct_mt", "max", "high_pct_mt"),
    "max_pct_ribo": ("pct_ribo", "max", "high_pct_ribo"),
    "max_pct_hb": ("pct_hb", "max", "high_pct_hb"),
}
GENE_SET_OF_METRIC = {"pct_mt": "mt", "pct_ribo": "ribo", "pct_hb": "hb"}


@dataclass(frozen=True)
class QCRules:
    """Thresholds for one dataset. ``None`` means "this rule is not applied"."""

    min_total_counts: float | None = None
    max_total_counts: float | None = None
    min_genes_detected: float | None = None
    max_genes_detected: float | None = None
    max_pct_mt: float | None = None
    max_pct_ribo: float | None = None
    max_pct_hb: float | None = None

    @classmethod
    def from_config(cls, qc_cfg: dict, dataset: str) -> "QCRules":
        """Merge ``qc.default`` with ``qc.per_dataset.<dataset>`` (the override wins)."""
        merged = dict(qc_cfg.get("default") or {})
        merged.update((qc_cfg.get("per_dataset") or {}).get(dataset) or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(merged) - known)
        if unknown:
            raise ValueError(f"qc rules for {dataset}: unknown keys {unknown}; known keys are {sorted(known)}")
        return cls(**merged)


def gene_set_masks(var: pd.DataFrame, patterns: dict | None = None) -> dict[str, np.ndarray]:
    """Boolean masks over features: mt (from gene identity), ribo and hb (from symbol), spike-ins."""
    patterns = patterns or DEFAULT_GENE_SET_PATTERNS
    symbol = var["symbol"].astype("string").fillna("")
    masks = {"mt": var["is_mt"].to_numpy(dtype=bool), "spike_in": var["is_spike_in"].to_numpy(dtype=bool)}
    for name, pattern in patterns.items():
        regex = re.compile(pattern)
        masks[name] = symbol.map(lambda s: bool(regex.match(s))).to_numpy(dtype=bool)
    return masks


def qc_metrics(adata, patterns: dict | None = None) -> pd.DataFrame:
    """Per-cell metrics on the registry unit in ``adata.X`` (never on a normalized matrix).

    Totals are over endogenous features only: ERCC spike-ins are not cellular RNA.
    Percentages for a gene set with zero members are NaN, never 0, because
    "not measured" is not "zero percent".
    """
    masks = gene_set_masks(adata.var, patterns)
    counts = sp.csr_matrix(adata.X, dtype=np.float64)
    endogenous = ~masks["spike_in"]
    endo = counts[:, endogenous]
    total = np.asarray(endo.sum(axis=1)).ravel()
    detected = np.asarray((endo > 0).sum(axis=1)).ravel()
    spike = np.asarray(counts[:, masks["spike_in"]].sum(axis=1)).ravel()
    frame = pd.DataFrame(
        {"total_counts": total, "n_genes_detected": detected.astype(np.int64)}, index=adata.obs_names.copy()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        for metric, gene_set in GENE_SET_OF_METRIC.items():
            members = masks[gene_set] & endogenous
            if members.sum() == 0:
                frame[metric] = np.nan
            else:
                frame[metric] = 100 * np.asarray(counts[:, members].sum(axis=1)).ravel() / total
        frame["pct_spike_in"] = 100 * spike / (total + spike) if masks["spike_in"].any() else np.nan
    frame.attrs["gene_set_sizes"] = {k: int(v.sum()) for k, v in masks.items()}
    return frame


def apply_qc(metrics: pd.DataFrame, rules: QCRules) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-cell verdicts, per-rule status table).

    Verdicts: ``qc_pass`` (bool) and ``qc_reason`` (failed rule labels joined by
    ';', empty when the cell passes). A rule whose gene set is absent from the
    dataset is recorded as ``not_applicable`` rather than silently passing.
    """
    failures = pd.DataFrame(index=metrics.index)
    status_rows = []
    for rule, (metric, direction, label) in RULES.items():
        threshold = getattr(rules, rule)
        values = metrics[metric]
        if threshold is None:
            status_rows.append({"rule": rule, "threshold": None, "status": "not_applied", "cells_failing": 0})
            continue
        if values.isna().all():
            status_rows.append({"rule": rule, "threshold": threshold, "status": "not_applicable",
                                "cells_failing": 0})
            continue
        failed = values < threshold if direction == "min" else values > threshold
        failed = failed.fillna(False)
        failures[label] = failed
        status_rows.append({"rule": rule, "threshold": threshold, "status": "applied",
                            "cells_failing": int(failed.sum())})
    if failures.shape[1]:
        reason = failures.apply(lambda row: ";".join(c for c in failures.columns if row[c]), axis=1)
    else:
        reason = pd.Series("", index=metrics.index)
    verdict = pd.DataFrame({"qc_pass": reason == "", "qc_reason": reason.astype(str)}, index=metrics.index)
    return verdict, pd.DataFrame(status_rows)


def retention_table(obs: pd.DataFrame, verdict: pd.DataFrame) -> pd.DataFrame:
    """Cells kept and removed by dataset, donor and harmonized class (proposal IV.D).

    Cells without a harmonized label (for example Muraro's author-QC failures)
    are grouped as '<no label>' so they are counted, not dropped from the report.
    """
    frame = pd.DataFrame(
        {
            "dataset": obs["dataset"].astype(str).to_numpy(),
            "donor": obs["donor"].astype(str).to_numpy(),
            "label_harmonized": obs["label_harmonized"].astype("string").fillna("<no label>").to_numpy(),
            "eligible_primary": obs["eligible_primary"].to_numpy(dtype=bool),
            "qc_pass": verdict["qc_pass"].to_numpy(dtype=bool),
        }
    )
    table = (
        frame.groupby(["dataset", "donor", "label_harmonized", "eligible_primary"], observed=True)
        .agg(n_cells=("qc_pass", "size"), n_pass=("qc_pass", "sum"))
        .reset_index()
    )
    table["n_fail"] = table["n_cells"] - table["n_pass"]
    table["fraction_retained"] = (table["n_pass"] / table["n_cells"]).round(4)
    return table.sort_values(["dataset", "donor", "label_harmonized"]).reset_index(drop=True)


def run_qc(adata, qc_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Metrics, verdicts and rule status for one loaded dataset."""
    dataset = adata.uns["annotbench"]["dataset"]
    metrics = qc_metrics(adata, qc_cfg.get("gene_sets"))
    verdict, status = apply_qc(metrics, QCRules.from_config(qc_cfg, dataset))
    status.insert(0, "dataset", dataset)
    sizes = metrics.attrs["gene_set_sizes"]
    status["genes_in_set"] = status["rule"].map(
        lambda r: sizes.get(GENE_SET_OF_METRIC.get(RULES[r][0], ""), np.nan)
    )
    return metrics, verdict, status
