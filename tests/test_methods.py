"""Tests for the method adapters (Week 2 guide, Steps 3 and 4)."""
import unittest

import numpy as np

from annotbench.methods import IneligibleInput, make_method
from annotbench.methods.base import TrainingData, check_output, donor_folds
from annotbench.methods.pca_knn import PcaKnn, select_k, vote_fractions


def clusters(n_per_class=20, seed=0, classes=("alpha", "beta", "gamma")):
    """Well-separated blobs in 5 dimensions, with 4 'donors' interleaved."""
    rng = np.random.default_rng(seed)
    reps, labels = [], []
    for k, name in enumerate(classes):
        centre = np.zeros(5)
        centre[k] = 20.0
        reps.append(centre + rng.normal(size=(n_per_class, 5)))
        labels += [name] * n_per_class
    rep = np.vstack(reps)
    groups = np.array([f"d{i % 4}" for i in range(len(labels))])
    return rep, np.array(labels), groups


def training(rep, labels, groups):
    ids = np.array([f"cell{i:04d}" for i in range(len(labels))])
    return TrainingData(rep, labels, ids, groups, lambda fit, apply: (rep[fit], rep[apply]))


class Knn(unittest.TestCase):
    def test_vote_fractions_sum_to_one_and_columns_follow_vocabulary(self):
        rep, labels, groups = clusters()
        knn = PcaKnn(k=5)
        knn.fit(training(rep, labels, groups), seed=0)
        out = knn.predict(rep + 0.1)
        np.testing.assert_allclose(out.scores.sum(axis=1), 1.0)
        self.assertEqual(out.vocabulary, ["alpha", "beta", "gamma"])
        self.assertEqual(out.score_type, "vote_fraction")
        check_output(out, len(rep))

    def test_separated_clusters_are_classified_perfectly(self):
        rep, labels, groups = clusters()
        query, truth, _ = clusters(seed=1)
        knn = PcaKnn(k=7)
        knn.fit(training(rep, labels, groups), seed=0)
        self.assertTrue(np.all(knn.predict(query).predicted_label == truth))

    def test_tie_goes_to_first_class_in_vocabulary(self):
        votes = vote_fractions(np.array([[1, 0], [0, 1]]), 2)
        np.testing.assert_array_equal(votes, [[0.5, 0.5], [0.5, 0.5]])
        rep = np.array([[0.0], [2.0]])
        knn = PcaKnn(k=2)
        knn.fit(TrainingData(rep, np.array(["zeta", "alpha"]), np.array(["a", "b"]), np.array(["d", "d"]),
                             lambda f, a: (rep[f], rep[a])), seed=0)
        self.assertEqual(knn.predict(np.array([[1.0]])).predicted_label[0], "alpha")

    def test_k_larger_than_training_set_is_ineligible(self):
        rep, labels, groups = clusters(n_per_class=2)
        with self.assertRaises(IneligibleInput):
            PcaKnn(k=50).fit(training(rep, labels, groups), seed=0)

    def test_k_selection_uses_donor_folds(self):
        rep, labels, groups = clusters()
        best, table = select_k(training(rep, labels, groups), [1, 3, 5])
        self.assertIn(best, [1, 3, 5])
        self.assertEqual(sorted(table["held_out_donor"].unique()), ["d0", "d1", "d2", "d3"])
        self.assertEqual(best, 1)  # every k is perfect here, so the smallest wins the tie


class Svm(unittest.TestCase):
    def test_scores_follow_vocabulary_and_sum_to_one(self):
        rep, labels, groups = clusters()
        svm = make_method("linear_svm", {"representation": "pca"})
        svm.fit(training(rep, labels, groups), seed=0)
        out = svm.predict(rep)
        self.assertEqual(out.vocabulary, ["alpha", "beta", "gamma"])
        self.assertEqual(out.score_type, "calibrated_probability")
        np.testing.assert_allclose(out.scores.sum(axis=1), 1.0)
        check_output(out, len(rep))

    def test_probability_calibration_only_sees_training_cells(self):
        rep, labels, groups = clusters()
        data = training(rep, labels, groups)
        svm = make_method("linear_svm", {"representation": "pca"})
        svm.fit(data, seed=0)
        self.assertTrue(set(svm.calibration_fit_ids) <= set(data.cell_ids))
        self.assertGreater(len(svm.calibration_fit_ids), 0)

    def test_uncalibrated_svm_reports_decision_scores(self):
        rep, labels, groups = clusters()
        svm = make_method("linear_svm", {"representation": "pca", "probability_calibration": "none"})
        svm.fit(training(rep, labels, groups), seed=0)
        self.assertEqual(svm.predict(rep).score_type, "decision_score")

    def test_unit_check(self):
        with self.assertRaises(IneligibleInput):
            make_method("pca_knn", {}).check_units({"x": "log_normalized"})


class Folds(unittest.TestCase):
    def test_donor_folds_hold_out_one_donor_each(self):
        folds = donor_folds(np.array(["b", "a", "b", "c"]))
        self.assertEqual([f[0] for f in folds], ["a", "b", "c"])
        self.assertEqual(list(folds[1][2]), [0, 2])
        self.assertEqual(donor_folds(np.array(["only", "only"])), [])


if __name__ == "__main__":
    unittest.main()
