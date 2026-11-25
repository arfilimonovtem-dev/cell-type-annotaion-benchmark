"""Tests for the scenario runner (Week 2 guide, Steps 5 and 6) on toy data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

from annotbench import methods
from annotbench.methods.pca_knn import PcaKnn
from annotbench.provenance import read_tsv
from annotbench.runner import load_pilot_config, run_pilot
from tests.toydata import CLASSES, toy_pair


def toy_config(folder: Path, **overrides) -> Path:
    label_map = folder / "label_map.csv"
    rows = [{"dataset": d, "original_label": c, "harmonized_label": c, "cl_id": f"CL:{i:07d}",
             "cl_id_verified": "false", "evidence": "toy", "ambiguity": "none", "eligible_primary": "true",
             "exclusion_reason": ""} for d in ("toyref", "toyqry") for i, c in enumerate(CLASSES)]
    pd.DataFrame(rows).to_csv(label_map, index=False)
    cfg = {
        "name": "toy", "label_map": str(label_map), "seeds": [0, 1], "split_version": "toy_v1",
        "label_task": "toy_v1",
        "qc": {"default": {"min_total_counts": 10}},
        "universe": {"name": "toy_universe", "datasets": ["toyref", "toyqry"]},
        "split_rules": {"cap_per_class": 20, "min_calibration_per_class": 5},
        "methods": {"pca_knn.k5": {"method": "pca_knn", "k": 5, "weights": "distance"},
                    "linear_svm": {"representation": "pca"}},
        "scenarios": [
            {"exp": "e1", "reference": "toyref", "query": "toyref", "eval_donors": ["r4"], "calibration_donors": ["r1"]},
            {"exp": "e2", "reference": "toyref", "query": "toyqry", "calibration_donors": ["r1"]},
            {"exp": "e3", "reference": "toyref", "query": "toyqry", "calibration_donors": ["r1"], "remove": ["delta"]},
        ],
        "preprocess": {"n_hvg": 30, "n_pcs": 5, "hvg_watch_genes": ["G0", "G59"]},
    }
    cfg.update(overrides)
    path = folder / "pilot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def run_toy(out: Path, config_folder: Path, **kwargs):
    cfg = load_pilot_config(toy_config(config_folder))
    ref, qry = toy_pair()
    return cfg, run_pilot(cfg, out_root=out, adatas={"toyref": ref, "toyqry": qry}, **kwargs)


def record(out: Path, run_id: str) -> dict:
    return json.loads((out / "results" / "runs" / run_id / "run_record.json").read_text())


class BrokenMethod(PcaKnn):
    name = "broken"

    def fit(self, train, seed):
        raise RuntimeError("planted failure inside fit")


class Runner(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_every_run_succeeds_and_writes_its_files(self):
        _, index = run_toy(self.root / "out", self.root)
        self.assertEqual(len(index), 3 * 2 * 2)
        self.assertEqual(set(index["status"]), {"success"})
        run_dir = self.root / "out" / "results" / "runs" / index["run_id"].iloc[0]
        for name in ("predictions.tsv.gz", "scores.tsv.gz", "run_record.json", "fitted/hvg.tsv",
                     "fitted/provenance.json"):
            self.assertTrue((run_dir / name).exists(), name)

    def test_failure_inside_fit_still_writes_a_record(self):
        with mock.patch.dict(methods.REGISTRY, {"broken": BrokenMethod}):
            cfg = load_pilot_config(toy_config(self.root, methods={"broken": {"k": 5}}))
            ref, qry = toy_pair()
            index = run_pilot(cfg, out_root=self.root / "out", adatas={"toyref": ref, "toyqry": qry},
                              exp_filter=["e2"])
        self.assertEqual(set(index["status"]), {"software_failure"})
        rec = record(self.root / "out", index["run_id"].iloc[0])
        self.assertEqual(rec["error_type"], "RuntimeError")
        self.assertIn("preprocessing", rec["stages"])  # it got that far, and the timing says so
        self.assertFalse((self.root / "out" / "results" / "runs" / rec["run_id"] / "predictions.tsv.gz").exists())

    def test_class_removal_reaches_vocabulary_and_scores(self):
        _, index = run_toy(self.root / "out", self.root, exp_filter=["e3"])
        for run_id in index["run_id"]:
            rec = record(self.root / "out", run_id)
            self.assertNotIn("delta", rec["vocabulary"])
            scores = read_tsv(self.root / "out" / "results" / "runs" / run_id / "scores.tsv.gz")
            self.assertNotIn("delta", scores.columns)
        split_id = index["split_id"].iloc[0]
        split = read_tsv(self.root / "out" / "splits" / "toy_v1" / f"{split_id}.tsv.gz")
        self.assertEqual(set(split["dataset"]), {"toyref"})
        delta = split[split["label_harmonized"] == "delta"]
        self.assertFalse(delta["role"].isin(["train", "train_reserve", "calibration"]).any())
        scenario_id = split_id.rsplit("__seed-", 1)[0]
        query = read_tsv(self.root / "out" / "query_sets" / "toy_v1" / f"{scenario_id}.tsv.gz")
        self.assertEqual(set(query.loc[query["label_harmonized"] == "delta", "role"]), {"evaluation_open_set"})

    def test_both_methods_receive_identical_inputs(self):
        _, index = run_toy(self.root / "out", self.root)
        keys = ["train_ids_sha256", "calibration_ids_sha256", "query_ids_sha256", "universe_sha256",
                "split_file_sha256", "hvg_sha256"]
        for split_id, group in index.groupby("split_id"):
            hashes = [{k: record(self.root / "out", r)["hashes"][k] for k in keys} for r in group["run_id"]]
            self.assertEqual(len(hashes), 2)
            self.assertEqual(hashes[0], hashes[1], split_id)

    def test_predictions_carry_no_true_labels(self):
        _, index = run_toy(self.root / "out", self.root, exp_filter=["e2"])
        predictions = read_tsv(self.root / "out" / "results" / "runs" / index["run_id"].iloc[0] / "predictions.tsv.gz")
        self.assertFalse({"label_harmonized", "label_original", "role"} & set(predictions.columns))
        self.assertEqual(set(predictions["cell_set"]), {"calibration", "query"})

    def test_same_config_and_seed_reproduce_byte_identical_outputs(self):
        _, index = run_toy(self.root / "a", self.root)
        run_toy(self.root / "b", self.root)
        for run_id in index["run_id"]:
            for name in ("predictions.tsv.gz", "scores.tsv.gz", "fitted/hvg.tsv", "fitted/pca_loadings.tsv.gz"):
                a = (self.root / "a" / "results" / "runs" / run_id / name).read_bytes()
                b = (self.root / "b" / "results" / "runs" / run_id / name).read_bytes()
                self.assertEqual(a, b, f"{run_id}/{name}")

    def test_second_seed_changes_training_cells_only(self):
        _, index = run_toy(self.root / "out", self.root, exp_filter=["e2"], method_filter=["pca_knn"])
        self.assertEqual(set(index["method"]), {"pca_knn.k5"})  # a family name selects its instances
        first, second = (record(self.root / "out", r)["hashes"] for r in sorted(index["run_id"]))
        self.assertNotEqual(first["train_ids_sha256"], second["train_ids_sha256"])
        self.assertEqual(first["calibration_ids_sha256"], second["calibration_ids_sha256"])
        self.assertEqual(first["query_ids_sha256"], second["query_ids_sha256"])
        self.assertEqual(first["universe_sha256"], second["universe_sha256"])

    def test_parallel_jobs_give_identical_bytes(self):
        _, index = run_toy(self.root / "serial", self.root, exp_filter=["e2", "e3"])
        run_toy(self.root / "parallel", self.root, exp_filter=["e2", "e3"], jobs=3)
        for run_id in index["run_id"]:
            for name in ("predictions.tsv.gz", "scores.tsv.gz", "fitted/pca_loadings.tsv.gz"):
                a = (self.root / "serial" / "results" / "runs" / run_id / name).read_bytes()
                b = (self.root / "parallel" / "results" / "runs" / run_id / name).read_bytes()
                self.assertEqual(a, b, f"{run_id}/{name}")

    def test_hvg_report_is_written_and_watched_genes_reported(self):
        _, index = run_toy(self.root / "out", self.root, exp_filter=["e2"], method_filter=["pca_knn"])
        run_dir = self.root / "out" / "results" / "runs" / index["run_id"].iloc[0]
        report = json.loads((run_dir / "fitted" / "hvg_report.json").read_text())
        self.assertEqual({g["symbol"] for g in report["watch_genes"]}, {"G0", "G59"})
        rec = record(self.root / "out", index["run_id"].iloc[0])
        self.assertIn("hvg_anomalies", rec)
        self.assertIsNotNone(rec["hvg_query_coverage"])
        self.assertTrue((self.root / "out" / "results" / "hvg_anomalies.tsv").exists())

    def test_evaluate_scores_every_run(self):
        from annotbench.metrics import evaluate_pilot

        _, index = run_toy(self.root / "out", self.root)
        tables = evaluate_pilot(self.root / "out", "toy_v1")
        summary = tables["run_summary"]
        self.assertEqual(len(summary), len(index))
        self.assertTrue((summary["accuracy"] > 0.9).all())  # toy classes are cleanly separated
        e3 = summary[summary["exp"] == "e3"]
        self.assertTrue((e3["n_unseen"] > 0).all())
        self.assertTrue(e3["balanced_rejection_accuracy"].notna().all())
        self.assertTrue((summary["calibration_frr"] <= 0.05).all())


if __name__ == "__main__":
    unittest.main()
