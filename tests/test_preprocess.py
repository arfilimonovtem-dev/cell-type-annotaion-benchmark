"""Tests for the preprocessor (Week 2 guide, Step 2): the frozen ruler."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from annotbench.preprocess import (
    PreprocessConfig, PreprocessError, Preprocessor, hvg_diagnostics, hvg_query_coverage, lognorm, rank_genes,
    resolve_target_sums, select_hvg,
)
from annotbench.provenance import sha256_ids
from tests.toydata import make_toy

CFG = PreprocessConfig(n_hvg=30, n_pcs=5, clip_max=10)


def toy_counts():
    adata = make_toy("toyref", ["r1", "r2", "r3", "r4"], seed=5)
    genes = adata.var["ncbi_gene_id"].astype(str).tolist()
    return adata, sp.csr_matrix(adata.X), genes, np.asarray(adata.obs_names)


def fitted(counts, genes, ids, cfg=CFG, seed=0):
    return Preprocessor(cfg, seed).fit(counts, genes, ids, "toyref", {"toyref": 10000.0})


class Stateless(unittest.TestCase):
    def test_lognorm_is_pure_and_per_cell(self):
        _, counts, _, _ = toy_counts()
        before = counts.copy()
        whole = lognorm(counts, 10000).toarray()
        alone = lognorm(counts[3:4], 10000).toarray()
        np.testing.assert_array_equal(whole[3:4], alone)
        self.assertEqual((counts != before).nnz, 0)

    def test_dataset_median_constants(self):
        sums = resolve_target_sums(PreprocessConfig(target_sum="dataset_median"),
                                   {"a": np.array([1.0, 3.0, 5.0]), "b": np.array([10.0, 20.0])})
        self.assertEqual(sums, {"a": 3.0, "b": 15.0})
        self.assertEqual(resolve_target_sums(CFG, {"a": np.array([1.0])}), {"a": 10000.0})


class FrozenRuler(unittest.TestCase):
    def test_extra_cells_do_not_change_the_ruler(self):
        _, counts, genes, ids = toy_counts()
        train = np.arange(0, 100)
        a = fitted(counts[train], genes, ids[train])
        b = fitted(counts[train], genes, ids[train])
        c = fitted(counts, genes, ids)  # all cells: a different ruler is expected
        np.testing.assert_array_equal(a.pca_components, b.pca_components)
        self.assertEqual(list(a.hvg["ncbi_gene_id"]), list(b.hvg["ncbi_gene_id"]))
        self.assertNotEqual(a.provenance["cell_ids_sha256"], c.provenance["cell_ids_sha256"])

    def test_cell_output_does_not_depend_on_batch_mates(self):
        _, counts, genes, ids = toy_counts()
        prep = fitted(counts[:100], genes, ids[:100])
        batch = prep.transform(counts[100:], genes, "toyref")["pca"]
        alone = prep.transform(counts[120:121], genes, "toyref")["pca"]
        np.testing.assert_allclose(batch[20:21], alone, rtol=0, atol=1e-12)

    def test_raw_counts_unchanged_after_fit_and_transform(self):
        _, counts, genes, ids = toy_counts()
        before = counts.copy()
        prep = fitted(counts, genes, ids)
        prep.transform(counts, genes, "toyref")
        self.assertEqual((counts != before).nnz, 0)

    def test_same_seed_same_saved_files(self):
        _, counts, genes, ids = toy_counts()
        names = ["hvg.tsv", "scaler.tsv", "pca_loadings.tsv.gz", "pca_center.tsv", "recipe.json"]
        contents = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as folder:
                fitted(counts, genes, ids).save(folder)
                contents.append([(Path(folder) / n).read_bytes() for n in names])
        self.assertEqual(contents[0], contents[1])

    def test_provenance_hash_equals_training_ids(self):
        _, counts, genes, ids = toy_counts()
        train = np.arange(10, 90)
        prep = fitted(counts[train], genes, ids[train])
        self.assertEqual(prep.provenance["cell_ids_sha256"], sha256_ids(ids[train][::-1]))
        self.assertEqual(prep.provenance["roles_used"], ["train"])

    def test_constant_gene_gets_unit_sd_not_nan(self):
        _, counts, genes, ids = toy_counts()
        dense = counts.toarray()
        dense[:, 0] = 0  # measured but unexpressed in every training cell: sd = 0 after normalization
        cfg = PreprocessConfig(n_hvg=len(genes), n_pcs=5, hvg_flavor="variance")
        prep = fitted(sp.csr_matrix(dense), genes, ids, cfg)
        row = prep.scaler.set_index("ncbi_gene_id").loc[genes[0]]
        self.assertTrue(row["sd_was_zero"])
        self.assertEqual(row["sd"], 1.0)
        self.assertTrue(np.isfinite(prep.transform(sp.csr_matrix(dense), genes, "toyref")["pca"]).all())

    def test_drop_rule_removes_constant_gene(self):
        _, counts, genes, ids = toy_counts()
        dense = counts.toarray()
        dense[:, 0] = 0
        cfg = PreprocessConfig(n_hvg=len(genes), n_pcs=5, hvg_flavor="variance", zero_sd_rule="drop")
        prep = fitted(sp.csr_matrix(dense), genes, ids, cfg)
        self.assertNotIn(genes[0], set(prep.hvg["ncbi_gene_id"]))


class SaveLoad(unittest.TestCase):
    def test_loaded_preprocessor_transforms_identically(self):
        _, counts, genes, ids = toy_counts()
        prep = fitted(counts[:100], genes, ids[:100])
        with tempfile.TemporaryDirectory() as folder:
            prep.save(folder)
            again = Preprocessor.load(folder, genes)
        for key in ("scaled_hvg", "pca"):
            np.testing.assert_array_equal(prep.transform(counts[100:], genes, "toyref")[key],
                                          again.transform(counts[100:], genes, "toyref")[key])

    def test_wrong_gene_order_is_refused(self):
        _, counts, genes, ids = toy_counts()
        prep = fitted(counts, genes, ids)
        with self.assertRaises(PreprocessError):
            prep.transform(counts, list(reversed(genes)), "toyref")

    def test_unknown_dataset_constant_is_refused(self):
        _, counts, genes, ids = toy_counts()
        with self.assertRaises(PreprocessError):
            fitted(counts, genes, ids).transform(counts, genes, "somewhere_else")


class Hvg(unittest.TestCase):
    def test_ties_broken_by_gene_id(self):
        matrix = sp.csr_matrix(np.tile(np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]), (5, 1)))
        table = select_hvg(matrix, ["40", "3", "20", "100"], 2, "variance")
        self.assertEqual(list(table["ncbi_gene_id"]), ["3", "20"])


class HvgReport(unittest.TestCase):
    def setUp(self):
        # 6 genes, 40 cells, 2 donors. g1 lives in donor b only; g5 is all zero; g4 is detected in 3 cells.
        rng = np.random.default_rng(0)
        x = rng.poisson(3.0, size=(40, 6)).astype(float)
        x[:20, 1] = 0.0
        x[:, 5] = 0.0
        x[:, 4] = 0.0
        x[:3, 4] = 50.0
        self.counts = sp.csr_matrix(x)
        self.genes = ["1", "2", "3", "4", "5", "6"]
        self.groups = np.array(["a"] * 20 + ["b"] * 20)
        self.lognormed = lognorm(self.counts, 1000.0)
        self.ranking = rank_genes(self.lognormed, self.genes, 3, "variance")

    def test_ranking_covers_every_gene_and_select_hvg_is_its_head(self):
        self.assertEqual(list(self.ranking["rank"]), [1, 2, 3, 4, 5, 6])
        self.assertEqual(list(select_hvg(self.lognormed, self.genes, 3, "variance")["ncbi_gene_id"]),
                         list(self.ranking["ncbi_gene_id"].head(3)))

    def test_anomalies_are_reported(self):
        report = hvg_diagnostics(self.lognormed, self.counts, self.genes, self.ranking, 3,
                                 watch={"3": "WATCHED", "999": "ABSENT"}, groups=self.groups)
        self.assertEqual(report["n_genes_all_zero_in_train"], 1)
        self.assertIn("5", report["low_detection_hvgs"])          # rank-1 gene seen in 3 cells only
        self.assertIn("2", report["donor_dominated_hvgs"])        # expressed in donor b only
        watched = {w["symbol"]: w for w in report["watch_genes"]}
        self.assertFalse(watched["ABSENT"]["in_universe"])
        self.assertIn("rank", watched["WATCHED"])
        self.assertTrue(report["anomalies"])

    def test_query_coverage_flags_undetected_hvgs(self):
        hvg = list(self.ranking["ncbi_gene_id"].head(3))
        query = sp.csr_matrix(np.zeros((5, 6)))
        coverage = hvg_query_coverage(query, self.genes, hvg)
        self.assertEqual(coverage["n_hvg_undetected"], 3)
        self.assertIsNotNone(coverage["anomaly"])
        self.assertIsNone(hvg_query_coverage(self.counts, self.genes, hvg[:1])["anomaly"])


if __name__ == "__main__":
    unittest.main()
