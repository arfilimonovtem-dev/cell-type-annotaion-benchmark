"""Tests for the pilot metrics (proposal VII.A, VII.C)."""
import unittest

import numpy as np

import pandas as pd

from annotbench.metrics import (
    auroc_low_score, class_balanced_rate, common_threshold, expected_calibration_error, multiclass_brier, query_token,
)


class Threshold(unittest.TestCase):
    def test_rejects_at_most_five_percent(self):
        scores = np.linspace(0.01, 1.0, 100)
        threshold, frr = common_threshold(scores)
        self.assertLessEqual(frr, 0.05)
        self.assertAlmostEqual(frr, 0.05)
        self.assertAlmostEqual(threshold, 0.06)

    def test_ties_are_kept_together_conservatively(self):
        # 10% of cells share the lowest score: rejecting them would exceed 5%, so none are rejected.
        scores = np.array([0.2] * 10 + [0.9] * 90)
        threshold, frr = common_threshold(scores)
        self.assertEqual((threshold, frr), (0.2, 0.0))

    def test_empty_or_missing_scores(self):
        self.assertTrue(np.isnan(common_threshold([])[0]))
        self.assertEqual(common_threshold([np.nan, 0.5, 0.7])[1], 0.0)


class Ece(unittest.TestCase):
    def test_perfectly_calibrated_bins_give_zero(self):
        confidence = np.array([0.75] * 4)
        correct = np.array([True, True, True, False])
        self.assertAlmostEqual(expected_calibration_error(confidence, correct), 0.0)

    def test_overconfidence_is_measured(self):
        self.assertAlmostEqual(expected_calibration_error(np.array([0.95] * 10), np.array([True] * 5 + [False] * 5)),
                               0.45)

    def test_confidence_one_falls_in_the_last_bin(self):
        self.assertAlmostEqual(expected_calibration_error(np.array([1.0]), np.array([True])), 0.0)


class Auroc(unittest.TestCase):
    def test_perfect_separation_and_ties(self):
        self.assertEqual(auroc_low_score([0.1, 0.2], [0.8, 0.9]), 1.0)
        self.assertEqual(auroc_low_score([0.8, 0.9], [0.1, 0.2]), 0.0)
        self.assertEqual(auroc_low_score([0.5], [0.5]), 0.5)
        self.assertTrue(np.isnan(auroc_low_score([], [0.5])))


class BrierAndBalance(unittest.TestCase):
    def test_brier_perfect_and_uniform(self):
        vocab = ["a", "b"]
        self.assertEqual(multiclass_brier(np.array([[1.0, 0.0], [0.0, 1.0]]), vocab, ["a", "b"]), 0.0)
        self.assertAlmostEqual(multiclass_brier(np.array([[0.5, 0.5]]), vocab, ["a"]), 0.5)

    def test_class_balanced_rate_weights_classes_equally(self):
        flags = pd.Series([True, False, False, False, True])
        classes = pd.Series(["big", "big", "big", "big", "small"])
        self.assertAlmostEqual(class_balanced_rate(flags, classes), (0.25 + 1.0) / 2)


class Tokens(unittest.TestCase):
    def test_query_token(self):
        self.assertEqual(query_token("exp-e1__ref-baron__qry-baron.human4__task-t__pert-none__seed-00"),
                         "baron.human4")


if __name__ == "__main__":
    unittest.main()
