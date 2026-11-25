"""Tests for the split step (Week 2 guide, Step 0).

Toy-data trap from the guide: with 12 cells per donor and class, a cap of 500
never triggers and the seed test would pass even if the seed did nothing. So the
cap here is 5.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from annotbench.splits import (
    QUERY_COLUMNS, ROLES, Scenario, SplitError, SplitRules, make_query_set, make_split, read_query_set, read_split,
    support_restrictions, train_classes_of, validate_split, write_query_set, write_split,
)
from tests.toydata import make_toy, toy_pair, with_qc_columns

RULES = SplitRules(cap_per_class=5, min_calibration_per_class=20, min_train_per_class=20, min_eval_per_class=10)
E2 = Scenario("e2", "toyref", "toyqry", calibration_donors=("r2",), label_task="toy_v1")
E1 = Scenario("e1", "toyref", "toyref", calibration_donors=("r2",), eval_donors=("r4",), label_task="toy_v1")
E3 = Scenario("e3", "toyref", "toyqry", calibration_donors=("r2",), remove=("delta",), label_task="toy_v1")
E3W = Scenario("e3", "toyref", "toyref", calibration_donors=("r2",), eval_donors=("r4",), remove=("delta",),
               label_task="toy_v1")


def pair_obs():
    ref, qry = toy_pair()
    return pd.concat([with_qc_columns(ref), with_qc_columns(qry)])


def query_set_for(obs, scenario, seed=0, rules=RULES):
    split = make_split(obs, scenario, rules, seed)
    return make_query_set(obs, scenario, train_classes_of(split), rules)


class Roles(unittest.TestCase):
    def test_every_reference_cell_has_exactly_one_known_role(self):
        obs = pair_obs()
        split = make_split(obs, E2, RULES, seed=0)
        self.assertEqual(len(split), int((obs["dataset"] == "toyref").sum()))
        self.assertFalse(split.duplicated(["dataset", "cell_id"]).any())
        self.assertTrue(set(split["role"]) <= set(ROLES))

    def test_no_donor_in_both_train_and_calibration(self):
        for scenario in (E1, E2, E3):
            split = make_split(pair_obs(), scenario, RULES, seed=0)
            train = set(split.loc[split["role"].isin(["train", "train_reserve"]), "donor"])
            calib = set(split.loc[split["role"] == "calibration", "donor"])
            self.assertEqual(train & calib, set(), scenario.exp)

    def test_e1_evaluation_donor_gives_no_reference_cell(self):
        split = make_split(pair_obs(), E1, RULES, seed=0)
        r4 = split[split["donor"] == "r4"]
        self.assertEqual(set(r4["role"]), {"evaluation"})
        self.assertNotIn("toyqry", set(split["dataset"]))

    def test_query_dataset_never_enters_the_split(self):
        for scenario in (E2, E3):
            split = make_split(pair_obs(), scenario, RULES, seed=0)
            self.assertEqual(set(split["dataset"]), {"toyref"}, scenario.exp)

    def test_query_cells_of_trained_classes_are_evaluation(self):
        obs = pair_obs()
        query = query_set_for(obs, E2)
        self.assertEqual(list(query.columns), QUERY_COLUMNS)
        self.assertEqual(len(query), int((obs["dataset"] == "toyqry").sum()))
        self.assertEqual(set(query["role"]), {"evaluation"})
        self.assertEqual(set(query["role_reason"]), {"query_dataset"})

    def test_query_set_does_not_depend_on_the_seed(self):
        obs = pair_obs()
        pd.testing.assert_frame_equal(query_set_for(obs, E2, seed=0), query_set_for(obs, E2, seed=1))

    def test_within_dataset_scenario_has_no_query_set(self):
        with self.assertRaises(SplitError):
            make_query_set(pair_obs(), E1, ["alpha", "beta", "delta"], RULES)

    def test_inner_fold_is_the_donor_on_training_cells_only(self):
        split = make_split(pair_obs(), E2, RULES, seed=0)
        training = split["role"].isin(["train", "train_reserve"])
        self.assertTrue((split.loc[training, "inner_fold"] == split.loc[training, "donor"]).all())
        self.assertTrue((split.loc[~training, "inner_fold"] == "").all())


class Seeds(unittest.TestCase):
    def test_same_seed_same_file_bytes(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            paths = [write_split(make_split(pair_obs(), E2, RULES, 7), RULES, folder, "v1", E2.split_id(7))
                     for folder in (a, b)]
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_other_seed_changes_only_train_versus_reserve(self):
        first, second = (make_split(pair_obs(), E2, RULES, s) for s in (0, 1))
        self.assertFalse(first["role"].equals(second["role"]), "the seed did nothing")
        changed = first["role"] != second["role"]
        self.assertTrue(set(first.loc[changed, "role"]) | set(second.loc[changed, "role"]) <= {"train", "train_reserve"})

    def test_row_order_does_not_change_the_draw(self):
        obs = pair_obs()
        shuffled = obs.sample(frac=1.0, random_state=3)
        pd.testing.assert_frame_equal(make_split(obs, E2, RULES, 0), make_split(shuffled, E2, RULES, 0))

    def test_cap_is_respected(self):
        split = make_split(pair_obs(), E2, RULES, 0)
        per_class = split.loc[split["role"] == "train", "label_harmonized"].value_counts()
        self.assertTrue((per_class == 5).all())


class ClassRemoval(unittest.TestCase):
    def test_removed_class_leaves_every_reference_role_but_stays_in_query(self):
        obs = pair_obs()
        split = make_split(obs, E3, RULES, 0)
        delta = split[split["label_harmonized"] == "delta"]
        self.assertFalse(delta["role"].isin(["train", "train_reserve", "calibration"]).any())
        self.assertEqual(set(delta["role_reason"]), {"class_removed:delta"})
        query = make_query_set(obs, E3, train_classes_of(split), RULES)
        query_delta = query[query["label_harmonized"] == "delta"]
        n_query_delta = int(((obs["dataset"] == "toyqry") & (obs["label_harmonized"] == "delta")).sum())
        self.assertEqual(len(query_delta), n_query_delta)
        self.assertEqual(set(query_delta["role"]), {"evaluation_open_set"})
        self.assertEqual(set(query_delta["role_reason"]), {"class_removed:delta"})

    def test_within_dataset_removal_makes_held_out_delta_open_set(self):
        split = make_split(pair_obs(), E3W, RULES, 0)
        delta = split[split["label_harmonized"] == "delta"]
        self.assertFalse(delta["role"].isin(["train", "train_reserve", "calibration"]).any())
        self.assertEqual(set(delta.loc[delta["donor"] == "r4", "role"]), {"evaluation_open_set"})
        self.assertEqual(set(split.loc[(split["donor"] == "r4") & (split["label_harmonized"] != "delta"), "role"]),
                         {"evaluation"})

    def test_split_id_names_the_perturbation(self):
        self.assertEqual(E3.split_id(0), "exp-e3__ref-toyref__qry-toyqry__task-toy_v1__pert-remove.delta__seed-00")
        self.assertEqual(E1.split_id(3), "exp-e1__ref-toyref__qry-toyref.r4__task-toy_v1__pert-none__seed-03")


class Ineligible(unittest.TestCase):
    def test_exclusions_and_challenge_set(self):
        ref, qry = toy_pair()
        obs_q = with_qc_columns(qry)
        obs_q["eligible_primary"] = obs_q["eligible_primary"].astype(bool)
        first, second, third = obs_q.index[:3]
        obs_q.loc[first, ["eligible_primary", "exclusion_reason"]] = [False, "uncertain_label"]
        obs_q.loc[second, ["eligible_primary", "exclusion_reason"]] = [False, "author_qc_fail"]
        obs_q.loc[third, ["qc_pass", "qc_reason"]] = [False, "low_total_counts"]
        query = query_set_for(pd.concat([with_qc_columns(ref), obs_q]), E2).set_index("cell_id")
        self.assertEqual(query.loc[first, "role"], "challenge_unlabeled")
        self.assertEqual((query.loc[second, "role"], query.loc[second, "role_reason"]), ("excluded", "author_qc_fail"))
        self.assertEqual((query.loc[third, "role"], query.loc[third, "role_reason"]),
                         ("excluded", "qc_fail:low_total_counts"))

    def test_split_never_reads_expression(self):
        ref = make_toy("toyref", ["r1", "r2", "r3", "r4"], nan_x=True)
        qry = make_toy("toyqry", ["q1"], nan_x=True)
        self.assertTrue(np.isnan(ref.X.toarray()).all())
        obs = pd.concat([with_qc_columns(ref), with_qc_columns(qry)])
        split = make_split(obs, E2, RULES, 0)
        query = make_query_set(obs, E2, train_classes_of(split), RULES)
        self.assertEqual(len(split) + len(query), ref.n_obs + qry.n_obs)

    def test_missing_qc_columns_stop_the_split(self):
        ref, qry = toy_pair()
        with self.assertRaises(SplitError):
            make_split(pd.concat([ref.obs, qry.obs]), E2, RULES, 0)


class Validation(unittest.TestCase):
    def test_planted_query_cell_in_split_is_caught(self):
        split = make_split(pair_obs(), E2, RULES, 0)
        split.loc[split.index[0], "dataset"] = "toyqry"
        with self.assertRaises(SplitError):
            validate_split(split, E2)

    def test_planted_overlap_is_caught(self):
        split = make_split(pair_obs(), E2, RULES, 0)
        calib = split.index[split["role"] == "calibration"][0]
        split.loc[calib, "role"] = "train"
        split.loc[split.index[split["role"] == "train"][0], "donor"] = "r2"
        with self.assertRaises(SplitError):
            validate_split(split, E2)

    def test_bad_scenarios_are_refused(self):
        with self.assertRaises(SplitError):
            Scenario("e1", "a", "a", calibration_donors=("x",)).validate()  # no eval donor
        with self.assertRaises(SplitError):
            Scenario("e2", "a", "b", calibration_donors=()).validate()
        with self.assertRaises(SplitError):
            Scenario("e3", "a", "b", calibration_donors=("x",)).validate()  # nothing removed

    def test_restrictions_report_flags_small_calibration(self):
        split = make_split(pair_obs(), E2, RULES, 0)
        report = support_restrictions(split, RULES)
        flagged = set(report.loc[report["check"] == "calibration_support", "class"])
        self.assertEqual(flagged, {"alpha", "beta", "delta"})  # 12 calibration cells each, below 20

    def test_written_split_and_query_set_read_back(self):
        obs = pair_obs()
        split = make_split(obs, E2, RULES, 0)
        query = make_query_set(obs, E2, train_classes_of(split), RULES)
        with tempfile.TemporaryDirectory() as folder:
            path = write_split(split, RULES, folder, "v1", E2.split_id(0), query)
            back = read_split(path)
            self.assertEqual(list(back["role"]), list(split["role"]))
            self.assertTrue((Path(folder) / "v1" / "manifest.tsv").exists())
            qpath = write_query_set(query, Path(folder) / "queries", "v1", E2.scenario_id)
            self.assertEqual(list(read_query_set(qpath)["role"]), list(query["role"]))
            self.assertNotIn("seed", qpath.name)

    def test_restrict_policy_moves_weak_class_to_open_set(self):
        obs = pair_obs()
        rules = SplitRules(cap_per_class=None, min_train_per_class=40, support_policy="restrict")
        split = make_split(obs, E2, rules, 0)  # 3 train donors x 12 cells = 36 < 40 for every class
        self.assertFalse((split["role"] == "train").any())
        query = make_query_set(obs, E2, train_classes_of(split), rules)
        self.assertEqual(set(query["role"]), {"evaluation_open_set"})


if __name__ == "__main__":
    unittest.main()
