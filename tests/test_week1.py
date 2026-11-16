"""Tests for the Week 1 code, using a small invented dataset.

Each test plants one specific defect and asserts the pipeline notices it.
Run from the repo root:  python -m unittest discover -s tests -t .
"""
import gzip
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from annotbench import config
from annotbench.genes import repair_r_names
from annotbench.integrity import IntegrityError, fatal_failures, raise_if_failed, run_integrity_checks
from annotbench.labels import LabelMapError, attach_labels, harmonize_obs
from annotbench.loading import DatasetLoadError, load_dataset

GENES = ["INS", "GCG", "ACTB", "GAPDH", "HLA.DRA", "MADEUP1"]
DONORS = ["d1", "d2", "d3", "d4"]
CLASSES = ["alpha", "beta"]
CELLS_PER_DONOR_CLASS = 6


def registry_for_toy() -> dict:
    return {
        "gene_info": "gene_info.tsv.gz",
        "marker_checks": {"INS": "beta", "GCG": "alpha"},
        "datasets": {
            "toy": {
                "h5ad": "toy.h5ad",
                "expression": {"use_layer": "counts"},
                "layer_units": {"counts": "umi_counts"},
                "expected_shape": {"n_cells": 48, "n_features": 6},
                "expected_missing_labels": 0,
                "columns": {"donor": "donor", "label": "cell_type1"},
                "ids": {"cell_id_regex": r"^d\d+_c\d+$", "donor_from_cell_id": r"^(d\d+)_"},
                "features": {"symbol_column": None, "r_mangled_names": True, "spike_in_column": None},
                "metadata_totals": [{"axis": "cells", "column": "total_counts", "kind": "sum", "layer": "counts"}],
                "annotation_source": {"path": "annotation.csv", "id_column": "cell_id", "label_column": "label",
                                      "donor_column": "donor"},
            }
        },
    }


def label_map_for_toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "original_label": CLASSES,
            "harmonized_label": CLASSES,
            "eligible_primary": [True, True],
            "exclusion_reason": ["", ""],
        }
    )


def make_adata() -> ad.AnnData:
    names, donors, labels, rows = [], [], [], []
    number = 0
    for donor in DONORS:
        for label in CLASSES:
            for _ in range(CELLS_PER_DONOR_CLASS):
                number += 1
                names.append(f"{donor}_c{number:03d}")
                donors.append(donor)
                labels.append(label)
                marker = [90, 2] if label == "beta" else [2, 90]  # INS, GCG
                rows.append([*marker, 30 + number, 20, number % 5, 1])
    matrix = sp.csr_matrix(np.array(rows, dtype="float32"))
    obs = pd.DataFrame({"donor": donors, "cell_type1": labels}, index=pd.Index(names, name="cell_id"))
    obs["total_counts"] = np.asarray(matrix.sum(axis=1)).ravel()
    var = pd.DataFrame(index=pd.Index(GENES, name="gene_id"))
    adata = ad.AnnData(X=matrix, obs=obs, var=var, layers={"counts": matrix.copy()})
    adata.uns["X_source_assay"] = "counts"
    return adata


def write_gene_info(root: Path) -> None:
    lines = ["#tax_id\tGeneID\tSymbol\tSynonyms\tchromosome"]
    lines += [
        "9606\t1\tINS\t-\t11", "9606\t2\tGCG\t-\t2", "9606\t3\tACTB\t-\t7", "9606\t4\tGAPDH\t-\t12",
        "9606\t5\tHLA-DRA\t-\t6",
    ]
    with gzip.open(root / "gene_info.tsv.gz", "wt") as handle:
        handle.write("\n".join(lines) + "\n")


def write_annotation(root: Path, adata: ad.AnnData) -> None:
    pd.DataFrame(
        {"cell_id": adata.obs_names, "label": adata.obs["cell_type1"].values, "donor": adata.obs["donor"].values}
    ).to_csv(root / "annotation.csv", index=False)


class ToyCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.registry = registry_for_toy()
        self.cfg = self.registry["datasets"]["toy"]
        self.label_map = label_map_for_toy()
        write_gene_info(self.root)

    def save(self, adata, annotation_from=None):
        adata.write_h5ad(self.root / "toy.h5ad")
        write_annotation(self.root, annotation_from if annotation_from is not None else adata)

    def run_checks(self):
        adata = load_dataset("toy", self.root, self.registry)
        return adata, run_integrity_checks(adata, self.cfg, self.registry, self.label_map, self.root)

    def failed_names(self):
        return {check.name for check in fatal_failures(self.run_checks()[1])}


class CleanData(ToyCase):
    def test_clean_dataset_passes_every_fatal_check(self):
        self.save(make_adata())
        self.assertEqual(self.failed_names(), set())

    def test_loader_keeps_original_identifiers_and_layers(self):
        self.save(make_adata())
        adata = load_dataset("toy", self.root, self.registry)
        self.assertEqual(list(adata.var["feature_id_original"]), GENES)
        self.assertEqual(adata.var.loc["HLA.DRA", "symbol"], "HLA-DRA")
        self.assertEqual(adata.var.loc["HLA.DRA", "symbol_original"], "HLA.DRA")
        self.assertIn("counts", adata.layers)
        self.assertEqual(adata.uns["annotbench"]["assay_unit"], "umi_counts")


class PlantedDefects(ToyCase):
    def test_duplicate_cell_id(self):
        adata = make_adata()
        names = list(adata.obs_names)
        names[1] = names[0]
        adata.obs_names = names
        adata.obs.index.name = "cell_id"
        adata.write_h5ad(self.root / "toy.h5ad")
        write_annotation(self.root, make_adata())
        self.assertIn("duplicate_cell_ids", self.failed_names())

    def test_duplicate_feature_id(self):
        adata = make_adata()
        adata.var_names = ["INS", "INS", "ACTB", "GAPDH", "HLA.DRA", "MADEUP1"]
        self.save(adata)
        self.assertIn("duplicate_feature_ids", self.failed_names())

    def _set_value(self, value, layer_also=True):
        adata = make_adata()
        adata.X.data[0] = value
        if layer_also:
            adata.layers["counts"].data[0] = value
        self.save(adata)

    def test_nan_value(self):
        self._set_value(np.nan)
        self.assertIn("nonfinite_in_X", self.failed_names())

    def test_negative_value(self):
        self._set_value(-3)
        self.assertIn("negative_in_X", self.failed_names())

    def test_non_integer_in_declared_counts(self):
        self._set_value(1.5)
        self.assertIn("non_integer_in_X", self.failed_names())

    def test_transposed_matrix(self):
        transposed = make_adata().T
        transposed.uns["X_source_assay"] = "counts"
        self.cfg["columns"] = {"donor": "cell_type1", "label": "cell_type1"}
        transposed.obs["cell_type1"] = "alpha"
        transposed.var["donor"] = "x"
        transposed.write_h5ad(self.root / "toy.h5ad")
        write_annotation(self.root, make_adata())
        failed = self.failed_names()
        self.assertIn("shape_is_cells_by_features", failed)
        self.assertIn("cell_axis_holds_cell_ids", failed)

    def test_transposed_file_stops_with_a_readable_message(self):
        transposed = make_adata().T
        transposed.uns["X_source_assay"] = "counts"
        transposed.write_h5ad(self.root / "toy.h5ad")
        write_annotation(self.root, make_adata())
        with self.assertRaises(DatasetLoadError) as caught:
            load_dataset("toy", self.root, self.registry)
        self.assertIn("swapped", str(caught.exception))

    def test_metadata_shuffled_against_matrix(self):
        adata = make_adata()
        adata.obs["total_counts"] = np.roll(adata.obs["total_counts"].to_numpy(), 7)
        self.save(adata)
        self.assertIn("recorded_cells_total_counts_matches_matrix", self.failed_names())

    def test_annotation_missing_one_cell(self):
        adata = make_adata()
        self.save(adata)
        table = pd.read_csv(self.root / "annotation.csv").iloc[1:]
        table.to_csv(self.root / "annotation.csv", index=False)
        self.assertIn("annotation_ids_match_cell_ids", self.failed_names())

    def test_annotation_label_disagrees(self):
        adata = make_adata()
        self.save(adata)
        table = pd.read_csv(self.root / "annotation.csv")
        table.loc[0, "label"] = "beta" if table.loc[0, "label"] == "alpha" else "alpha"
        table.to_csv(self.root / "annotation.csv", index=False)
        self.assertIn("annotation_label_matches_cell", self.failed_names())

    def test_labels_swapped_with_valid_ids_is_caught_by_markers(self):
        adata = make_adata()
        adata.obs["cell_type1"] = adata.obs["cell_type1"].map({"alpha": "beta", "beta": "alpha"})
        self.save(adata)
        self.assertIn("marker_genes_peak_in_expected_class", self.failed_names())

    def test_donor_column_disagrees_with_cell_id(self):
        adata = make_adata()
        adata.obs["donor"] = np.roll(adata.obs["donor"].to_numpy(), 12)
        self.save(adata)
        self.assertIn("donor_column_matches_cell_id", self.failed_names())

    def test_unlisted_label_stops(self):
        adata = make_adata()
        adata.obs["cell_type1"] = adata.obs["cell_type1"].astype(str)
        adata.obs.iloc[0, adata.obs.columns.get_loc("cell_type1")] = "duct"
        self.save(adata)
        self.assertIn("label_map_lists_every_label", self.failed_names())

    def test_failure_raises_a_readable_stop(self):
        self._set_value(-3)
        _, results = self.run_checks()
        with self.assertRaises(IntegrityError) as caught:
            raise_if_failed("toy", results, "report.md")
        self.assertIn("INTEGRITY STOP", str(caught.exception))
        self.assertIn("negative_in_X", str(caught.exception))


class Paths(ToyCase):
    def test_missing_file_names_what_is_missing(self):
        with self.assertRaises(config.MissingInputError) as caught:
            load_dataset("toy", self.root, self.registry)
        self.assertIn("toy.h5ad", str(caught.exception))

    def test_data_root_comes_from_environment(self):
        with mock.patch.dict(os.environ, {config.DATA_ROOT_ENV: str(self.root)}):
            self.assertEqual(config.resolve_data_root(), self.root.resolve())

    def test_explicit_data_root_beats_environment(self):
        with tempfile.TemporaryDirectory() as other, mock.patch.dict(os.environ, {config.DATA_ROOT_ENV: other}):
            self.assertEqual(config.resolve_data_root(self.root), self.root.resolve())


class GeneNames(unittest.TestCase):
    def test_repair_only_when_ncbi_confirms(self):
        names = pd.Series(["HLA.DRA", "BACH1.IT1", "ACTB"], dtype="string")
        repaired, accepted = repair_r_names(names, {"HLA-DRA", "ACTB"})
        self.assertEqual(list(repaired), ["HLA-DRA", "BACH1.IT1", "ACTB"])
        self.assertEqual(list(accepted), [True, False, False])


class Labels(unittest.TestCase):
    def test_missing_map_row_stops_instead_of_giving_nan(self):
        obs = pd.DataFrame({"label_original": pd.Series(["alpha", "duct"], dtype="string")})
        with self.assertRaises(LabelMapError) as caught:
            harmonize_obs(obs, "toy", label_map_for_toy(), None)
        self.assertIn("duct", str(caught.exception))

    def test_unlabelled_cells_are_ineligible_with_the_declared_reason(self):
        obs = pd.DataFrame({"label_original": pd.Series(["alpha", pd.NA], dtype="string")})
        result = harmonize_obs(obs, "toy", label_map_for_toy(), "author_qc_fail")
        self.assertEqual(list(result["eligible_primary"]), [True, False])
        self.assertEqual(result["exclusion_reason"].iloc[1], "author_qc_fail")


if __name__ == "__main__":
    unittest.main()
