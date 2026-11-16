"""Integrity checks that run before any cell is assigned to a split.

Each check records what it saw and what it expected. All checks run first, so
one report shows every problem, and then the run stops if a fatal check failed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import require_file
from .labels import harmonize_obs, unknown_labels
from .loading import layer_names

FATAL = "fatal"
WARNING = "warning"
INTEGER_UNITS = {"umi_counts"}
MIN_CELLS_FOR_MARKER_CLASS = 5


class IntegrityError(RuntimeError):
    """A fatal integrity check failed, so later steps must not run."""


@dataclass
class Check:
    group: str
    name: str
    severity: str
    passed: bool | None  # None means the check could not run because an earlier one failed
    observed: str
    expected: str
    detail: str = ""


def _examples(values, limit: int = 5) -> str:
    items = [str(v) for v in list(values)[:limit]]
    return ", ".join(items)


def _values(matrix) -> np.ndarray:
    return matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()


def _axis_values(matrix, axis: str, kind: str) -> np.ndarray:
    along = 1 if axis == "cells" else 0
    if kind == "sum":
        return np.asarray(matrix.astype("float64").sum(axis=along)).ravel()
    return np.asarray((matrix > 0).sum(axis=along)).ravel()


def check_cell_ids(adata: ad.AnnData, cfg: dict) -> list[Check]:
    names = pd.Index(adata.obs_names.astype(str))
    duplicated = names[names.duplicated()]
    blank = names[names.str.strip().isin(["", "nan", "None"])]
    pattern = re.compile(cfg["ids"]["cell_id_regex"])
    malformed = [n for n in names if not pattern.match(n)]
    return [
        Check("cell_ids", "duplicate_cell_ids", FATAL, len(duplicated) == 0, str(len(duplicated)), "0",
              _examples(duplicated)),
        Check("cell_ids", "blank_cell_ids", FATAL, len(blank) == 0, str(len(blank)), "0"),
        Check("orientation", "cell_axis_holds_cell_ids", FATAL, not malformed, f"{len(malformed)} malformed",
              f"all match {cfg['ids']['cell_id_regex']}", _examples(malformed)),
    ]


def check_gene_ids(adata: ad.AnnData) -> list[Check]:
    names = pd.Index(adata.var_names.astype(str))
    duplicated = names[names.duplicated()]
    var = adata.var
    how = var["id_resolution"]
    ambiguous = var.index[how == "ambiguous_name"]
    unmatched = var.index[how == "unmatched"]
    shared = var.index[var["shares_gene_id"]]
    return [
        Check("gene_ids", "duplicate_feature_ids", FATAL, len(duplicated) == 0, str(len(duplicated)), "0",
              _examples(duplicated)),
        Check("gene_ids", "ambiguous_gene_names", WARNING, len(ambiguous) == 0, str(len(ambiguous)), "0",
              "Name points to several NCBI genes, so it receives no Gene ID. " + _examples(ambiguous)),
        Check("gene_ids", "unmatched_gene_names", WARNING, len(unmatched) == 0, str(len(unmatched)), "0",
              "Name is unknown to NCBI gene_info. " + _examples(unmatched)),
        Check("gene_ids", "features_sharing_a_gene_id", WARNING, len(shared) == 0, str(len(shared)), "0",
              "Several features resolve to one Gene ID. " + _examples(shared)),
    ]


def check_values(adata: ad.AnnData) -> list[Check]:
    units = adata.uns["annotbench"]["layer_units"]
    results: list[Check] = []
    matrices = [("X", adata.X, adata.uns["annotbench"]["assay_unit"])]
    matrices += [(f"layer {n}", adata.layers[n], units[n]) for n in layer_names(adata)]
    for label, matrix, unit in matrices:
        values = _values(matrix)
        nonfinite = int((~np.isfinite(values)).sum())
        negative = int((values < 0).sum())
        results.append(Check("values", f"nonfinite_in_{label}", FATAL, nonfinite == 0, str(nonfinite), "0"))
        results.append(Check("values", f"negative_in_{label}", FATAL, negative == 0, str(negative), "0"))
        if unit in INTEGER_UNITS:
            fractional = int((values != np.round(values)).sum())
            results.append(Check("values", f"non_integer_in_{label}", FATAL, fractional == 0, str(fractional), "0",
                                 f"unit is declared as {unit}"))
    zero_cells = int((_axis_values(adata.X, "cells", "sum") == 0).sum())
    results.append(Check("values", "cells_with_zero_total", WARNING, zero_cells == 0, str(zero_cells), "0"))
    missing_donors = int(adata.obs["donor"].isna().sum())
    results.append(Check("values", "cells_without_donor", FATAL, missing_donors == 0, str(missing_donors), "0"))
    return results


def check_orientation(adata: ad.AnnData, cfg: dict, markers: list[str]) -> list[Check]:
    expected = cfg["expected_shape"]
    shape_ok = (adata.n_obs, adata.n_vars) == (expected["n_cells"], expected["n_features"])
    symbols = set(adata.var["symbol"].dropna().astype(str))
    missing = [m for m in markers if m not in symbols]
    on_cell_axis = [m for m in markers if m in set(adata.obs_names.astype(str))]
    return [
        Check("orientation", "shape_is_cells_by_features", FATAL, shape_ok, f"{adata.n_obs} x {adata.n_vars}",
              f"{expected['n_cells']} x {expected['n_features']}"),
        Check("orientation", "feature_axis_holds_genes", FATAL, not missing and not on_cell_axis,
              f"{len(markers) - len(missing)} of {len(markers)} marker genes on the feature axis",
              f"all {len(markers)}", _examples(missing + on_cell_axis)),
    ]


def _check_layer_patterns(adata: ad.AnnData) -> list[Check]:
    results = []
    for name in layer_names(adata):
        left, right = adata.X != 0, adata.layers[name] != 0
        different = int((left != right).nnz)
        results.append(Check("expression_metadata", f"layer_{name}_same_nonzero_cells_as_X", FATAL, different == 0,
                             f"{different} positions differ", "0"))
    return results


def _check_recorded_totals(adata: ad.AnnData, cfg: dict) -> list[Check]:
    results = []
    for rule in cfg.get("metadata_totals", []):
        frame = adata.obs if rule["axis"] == "cells" else adata.var
        name = f"recorded_{rule['axis']}_{rule['column']}_matches_matrix"
        if rule["column"] not in frame:
            results.append(Check("expression_metadata", name, FATAL, False, "column is absent", f"column {rule['column']}"))
            continue
        computed = _axis_values(adata.layers[rule["layer"]], rule["axis"], rule["kind"])
        recorded = pd.to_numeric(frame[rule["column"]]).to_numpy(float)
        wrong = int((~np.isclose(computed, recorded, rtol=1e-6, atol=1e-6)).sum())
        results.append(Check("expression_metadata", name, FATAL, wrong == 0, f"{wrong} mismatches", "0"))
    return results


def _check_source_matrix(adata: ad.AnnData, cfg: dict, root: Path) -> list[Check]:
    rule = cfg.get("source_matrix")
    if not rule:
        return []
    path = require_file(root, rule["path"], "deposited expression table")
    table = pd.read_csv(path, sep=rule["sep"], index_col=0)
    rows, columns = table.index.astype(str), table.columns.astype(str)
    rows_ok = set(rows) == set(adata.var_names.astype(str)) and len(rows) == adata.n_vars
    columns_ok = set(columns) == set(adata.obs_names.astype(str)) and len(columns) == adata.n_obs
    results = [Check("orientation", "deposited_table_rows_are_features_and_columns_are_cells", FATAL,
                     rows_ok and columns_ok, f"table {len(rows)} rows x {len(columns)} columns",
                     f"{adata.n_vars} features x {adata.n_obs} cells")]
    if not (rows_ok and columns_ok):
        results.append(Check("expression_metadata", "deposited_table_totals_match_matrix", FATAL, None, "not run", ""))
        return results
    table.index, table.columns = rows, columns
    layer = adata.layers[rule["layer"]]
    per_cell = table[list(adata.obs_names.astype(str))].sum(axis=0).to_numpy(float)
    per_gene = table.loc[list(adata.var_names.astype(str))].sum(axis=1).to_numpy(float)
    wrong_cells = int((~np.isclose(_axis_values(layer, "cells", "sum"), per_cell, rtol=1e-4)).sum())
    wrong_genes = int((~np.isclose(_axis_values(layer, "features", "sum"), per_gene, rtol=1e-4, atol=1e-2)).sum())
    results.append(Check("expression_metadata", "deposited_table_totals_match_matrix", FATAL,
                         wrong_cells == 0 and wrong_genes == 0,
                         f"{wrong_cells} cell totals and {wrong_genes} gene totals differ", "0 and 0"))
    return results


def _check_umi_back_transform(adata: ad.AnnData, cfg: dict) -> list[Check]:
    """Confirm the derived UMI layer is what inverting the saturation correction gives."""
    rule = cfg.get("umi_back_transform")
    if not rule:
        return []
    corrected, umi = adata.layers[rule["from_layer"]], adata.layers[rule["to_layer"]]
    k = rule["k"]
    recovered = k * (1 - np.exp(-corrected.data.astype("float64") / k))
    rounded = np.round(recovered)
    is_integer = np.isclose(recovered, rounded, atol=1e-3)
    fraction = float(is_integer.mean())
    # A saturated entry (all k UMIs seen) would invert to infinity, so the deposited
    # table stores the value for k - 0.5. Those are the only expected non-integers.
    at_cap = np.isclose(recovered, k - 0.5, atol=1e-3)
    unexplained = int((~is_integer & ~at_cap).sum())
    same = corrected.nnz == umi.nnz and np.array_equal(rounded.astype("float32"), umi.data)
    return [
        Check("expression_metadata", "umi_recovery_gives_integers", FATAL,
              fraction >= rule["min_integer_fraction"] and unexplained == 0,
              f"{fraction:.6f} integer, {int(at_cap.sum())} at the saturation cap, {unexplained} unexplained",
              f">= {rule['min_integer_fraction']} integer and 0 unexplained"),
        Check("expression_metadata", "umi_layer_equals_recovered_values", FATAL, same, str(same), "True"),
    ]


def check_expression_metadata(adata: ad.AnnData, cfg: dict, root: Path) -> list[Check]:
    pattern = re.compile(cfg["ids"]["donor_from_cell_id"])
    from_id = adata.obs_names.astype(str).map(lambda n: m.group(1) if (m := pattern.match(n)) else None)
    disagree = int((np.asarray(from_id, dtype=object) != adata.obs["donor"].astype(str).to_numpy(dtype=object)).sum())
    results = [Check("expression_metadata", "donor_column_matches_cell_id", FATAL, disagree == 0,
                     f"{disagree} disagreements", "0")]
    results += _check_layer_patterns(adata)
    results += _check_recorded_totals(adata, cfg)
    results += _check_source_matrix(adata, cfg, root)
    results += _check_umi_back_transform(adata, cfg)
    return results


def check_annotation(adata: ad.AnnData, cfg: dict, root: Path, dataset: str, label_map: pd.DataFrame) -> list[Check]:
    source = cfg["annotation_source"]
    path = require_file(root, source["path"], f"{dataset} annotation table")
    columns = [source["id_column"], source["label_column"], source["donor_column"]]
    table = pd.read_csv(path, usecols=columns, dtype=str)
    ids = table[source["id_column"]]
    cells = pd.Index(adata.obs_names.astype(str))

    repeated = int(ids.duplicated().sum())
    only_matrix = cells.difference(pd.Index(ids))
    only_table = pd.Index(ids).difference(cells)
    aligned = repeated == 0 and len(only_matrix) == 0 and len(only_table) == 0
    results = [
        Check("annotation", "annotation_ids_are_unique", FATAL, repeated == 0, str(repeated), "0"),
        Check("annotation", "annotation_ids_match_cell_ids", FATAL, aligned,
              f"{len(only_matrix)} cells without annotation, {len(only_table)} annotations without a cell", "0 and 0",
              _examples(list(only_matrix) + list(only_table))),
    ]
    if aligned:
        table = table.set_index(source["id_column"]).loc[cells]
        prefix = cfg["columns"].get("donor_prefix", "")
        donor = (prefix + table[source["donor_column"]]).to_numpy(dtype=object)
        wrong_donor = int((donor != adata.obs["donor"].astype(str).to_numpy(dtype=object)).sum())
        loaded = adata.obs["label_original"].astype(object).fillna("<missing>").to_numpy(dtype=object)
        deposited = table[source["label_column"]].astype(object).fillna("<missing>").to_numpy(dtype=object)
        wrong_label = int((loaded != deposited).sum())
        results.append(Check("annotation", "annotation_donor_matches_cell", FATAL, wrong_donor == 0, str(wrong_donor), "0"))
        results.append(Check("annotation", "annotation_label_matches_cell", FATAL, wrong_label == 0, str(wrong_label), "0"))
    else:
        results.append(Check("annotation", "annotation_donor_matches_cell", FATAL, None, "not run", "0"))
        results.append(Check("annotation", "annotation_label_matches_cell", FATAL, None, "not run", "0"))

    missing = int(adata.obs["label_original"].isna().sum())
    results.append(Check("annotation", "cells_without_author_label", FATAL, missing == cfg["expected_missing_labels"],
                         str(missing), str(cfg["expected_missing_labels"])))
    unknown = unknown_labels(adata.obs, dataset, label_map)
    results.append(Check("annotation", "label_map_lists_every_label", FATAL, not unknown, f"{len(unknown)} unlisted", "0",
                         _examples(unknown)))
    return results


def check_marker_alignment(adata: ad.AnnData, cfg: dict, dataset: str, label_map: pd.DataFrame, markers: dict) -> list[Check]:
    """Each canonical marker should peak in its own cell type.

    Identifier checks cannot see a shuffle that keeps the IDs valid, while this
    biological check can. It only tests alignment and never assigns a label.
    """
    name = "marker_genes_peak_in_expected_class"
    if unknown_labels(adata.obs, dataset, label_map):
        return [Check("annotation", name, FATAL, None, "not run", "all markers")]
    obs = harmonize_obs(adata.obs, dataset, label_map, cfg["columns"].get("missing_label_means"))
    keep = obs["eligible_primary"].to_numpy()
    matrix = adata.X[keep]
    labels = obs.loc[keep, "label_harmonized"].astype(str).to_numpy()
    sizes = pd.Series(labels).value_counts()
    usable = set(sizes[sizes >= MIN_CELLS_FOR_MARKER_CLASS].index)
    totals = np.asarray(matrix.astype("float64").sum(axis=1)).ravel()
    symbols = adata.var["symbol"].astype(str).to_numpy()

    wrong, found = [], []
    for marker, expected in markers.items():
        positions = np.flatnonzero(symbols == marker)
        if positions.size != 1:
            wrong.append(f"{marker}: {positions.size} features")
            continue
        values = np.log1p(1e4 * matrix[:, int(positions[0])].toarray().ravel() / np.maximum(totals, 1))
        means = pd.Series(values).groupby(labels).mean()
        top = means[means.index.isin(usable)].idxmax()
        found.append(f"{marker}={top}")
        if top != expected:
            wrong.append(f"{marker}: highest in {top}, expected {expected}")
    return [Check("annotation", name, FATAL, not wrong, f"{len(markers) - len(wrong)} of {len(markers)} correct",
                  f"{len(markers)} of {len(markers)}", "; ".join(wrong or found))]


def run_integrity_checks(
    adata: ad.AnnData, cfg: dict, registry: dict, label_map: pd.DataFrame, root: Path
) -> list[Check]:
    dataset = adata.uns["annotbench"]["dataset"]
    markers = registry["marker_checks"]
    results: list[Check] = []
    results += check_cell_ids(adata, cfg)
    results += check_gene_ids(adata)
    results += check_values(adata)
    results += check_orientation(adata, cfg, list(markers))
    results += check_expression_metadata(adata, cfg, root)
    results += check_annotation(adata, cfg, root, dataset, label_map)
    results += check_marker_alignment(adata, cfg, dataset, label_map, markers)
    return results


def fatal_failures(results: list[Check]) -> list[Check]:
    return [r for r in results if r.severity == FATAL and r.passed is False]


def raise_if_failed(dataset: str, results: list[Check], report: str | Path) -> None:
    failed = fatal_failures(results)
    if failed:
        lines = [f"INTEGRITY STOP: {dataset} failed {len(failed)} fatal check(s)."]
        lines += [f"  - [{r.group}] {r.name}: saw {r.observed}, expected {r.expected}. {r.detail}".rstrip() for r in failed]
        lines.append(f"Full report: {report}")
        raise IntegrityError("\n".join(lines))


def results_as_records(results: list[Check]) -> list[dict]:
    return [asdict(r) for r in results]


def feature_issue_table(adata: ad.AnnData) -> pd.DataFrame:
    var = adata.var
    flagged = var["id_resolution"].isin(["ambiguous_name", "unmatched"]) | var["shares_gene_id"]
    columns = ["feature_id_original", "symbol_original", "symbol", "symbol_repaired", "ncbi_gene_id",
               "id_resolution", "shares_gene_id"]
    return var.loc[flagged, columns].sort_values(["id_resolution", "symbol"])
