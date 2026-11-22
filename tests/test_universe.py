"""Tests for the gene universe and alignment (Week 2 guide, Step 1)."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from annotbench.universe import UniverseError, UniversePolicy, align, build_universe, write_universe
from tests.toydata import N_GENES, make_toy, toy_pair


def column_hash(adata, column_name) -> str:
    position = list(adata.var_names).index(column_name)
    return hashlib.sha256(np.ascontiguousarray(adata.X[:, position].toarray()).tobytes()).hexdigest()


class Universe(unittest.TestCase):
    def test_intersection_sorted_by_numeric_id(self):
        ref, qry = toy_pair()
        universe, excluded = build_universe({"toyref": ref.var, "toyqry": qry.var}, UniversePolicy())
        ids = universe["ncbi_gene_id"].tolist()
        self.assertEqual(len(ids), N_GENES - 2)
        self.assertEqual(ids, sorted(ids, key=int))
        missing = excluded[excluded["reason"] == "not_in_every_dataset"]
        self.assertEqual(set(missing["dataset"]), {"toyref"})
        self.assertEqual(len(missing), 2)

    def test_matching_is_by_id_not_name(self):
        ref, qry = toy_pair()
        qry.var.loc["G7", "symbol"] = "OLDNAME7"  # same Gene ID, different symbol (like ACKR1 / DARC)
        universe, _ = build_universe({"toyref": ref.var, "toyqry": qry.var}, UniversePolicy())
        row = universe[universe["ncbi_gene_id"] == "1007"].iloc[0]
        self.assertEqual((row["symbol_toyref"], row["symbol_toyqry"]), ("G7", "OLDNAME7"))

    def test_policy_drops_unverified_shared_and_spike_ins_with_reasons(self):
        ref = make_toy("toyref", ["r1"], spike_ins=2)
        var = ref.var.copy()
        var.loc["G1", "id_resolution"] = "ambiguous_name"
        var.loc["G1", "ncbi_gene_id"] = pd.NA
        var.loc[["G2", "G3"], "shares_gene_id"] = True
        universe, excluded = build_universe({"toyref": var}, UniversePolicy())
        reasons = excluded.set_index("feature_id_original")["reason"]
        self.assertEqual(reasons["G1"], "ambiguous_name")
        self.assertEqual(reasons["G2"], "shares_gene_id")
        self.assertEqual(reasons["ERCC-00000"], "spike_in")
        self.assertNotIn("1001", set(universe["ncbi_gene_id"]))

    def test_unexpressed_gene_stays(self):
        ref, qry = toy_pair()
        ref.X[:, 10] = 0
        universe, _ = build_universe({"toyref": ref.var, "toyqry": qry.var}, UniversePolicy())
        self.assertIn("1010", set(universe["ncbi_gene_id"]))

    def test_files_are_byte_identical_across_runs(self):
        ref, qry = toy_pair()
        contents = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as folder:
                universe, excluded = build_universe({"toyref": ref.var, "toyqry": qry.var}, UniversePolicy())
                path = write_universe(universe, excluded, folder, "toy_v1", {"status": "test"})
                contents.append((path.read_bytes(), (Path(folder) / "toy_v1_excluded.tsv").read_bytes()))
        self.assertEqual(contents[0], contents[1])

    def test_sum_policy_is_not_silently_accepted(self):
        with self.assertRaises(NotImplementedError):
            UniversePolicy.from_config({"shared_id_policy": "sum"})


class Align(unittest.TestCase):
    def test_both_datasets_get_identical_columns_and_unchanged_counts(self):
        ref, qry = toy_pair()
        universe, _ = build_universe({"toyref": ref.var, "toyqry": qry.var}, UniversePolicy())
        ids = universe["ncbi_gene_id"].tolist()
        before = {g: column_hash(ref, f"G{int(g) - 1000}") for g in ids}
        a_ref, a_qry = align(ref, ids), align(qry, ids)
        self.assertEqual(list(a_ref.var_names), list(a_qry.var_names))
        self.assertEqual(list(a_ref.var_names), ids)
        self.assertTrue(all(before[g] == column_hash(a_ref, g) for g in ids))
        self.assertEqual(list(a_ref.var["feature_id_original"][:3]), ["G0", "G1", "G2"])

    def test_missing_universe_gene_raises_instead_of_filling_zeros(self):
        ref, qry = toy_pair()
        ids = [str(1000 + i) for i in range(N_GENES)]  # includes the 2 genes qry lacks
        with self.assertRaises(UniverseError) as caught:
            align(qry, ids)
        self.assertIn("absent", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
