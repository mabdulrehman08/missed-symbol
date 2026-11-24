import unittest

import numpy as np

from demo import (
    TEST_SEEDS, TUNE_SEEDS, Box, Sheet, baseline, drawing, match, metrics,
    second_pass,
)


class DemoTests(unittest.TestCase):
    def test_drawing_is_reproducible_and_splits_are_disjoint(self):
        first, truth = drawing(TEST_SEEDS[0])
        again, again_truth = drawing(TEST_SEEDS[0])
        np.testing.assert_array_equal(first, again)
        self.assertEqual(truth, again_truth)
        self.assertEqual(len(truth), len(set(truth)))
        self.assertTrue(set(TUNE_SEEDS).isdisjoint(TEST_SEEDS))

    def test_duplicate_box_is_wrong_and_matching_is_one_to_one(self):
        hit = Box(22, 20, .91)
        duplicate = Box(23, 20, .87)
        hits, wrong, remaining = match([hit, duplicate], [(20, 20), (80, 80)])
        self.assertEqual(hits, {0: hit})
        self.assertEqual(wrong, [duplicate])
        self.assertEqual(remaining, {1})

    def test_invisible_miss_has_no_intersecting_baseline_box(self):
        sheet = Sheet(0, [(30, 40), (90, 40), (150, 40)], [
            Box(30, 40, .92), Box(71, 40, .81), Box(125, 40, .80), Box(150, 40, .60)
        ], [])
        base = baseline(sheet, .75)
        self.assertEqual(base.missed, {1, 2})
        self.assertEqual(base.invisible, {2})
        self.assertEqual(len(base.wrong), 2)

    def test_review_only_adds_boxes_and_counts_wrong_additions(self):
        hit = Box(30, 30, .81)
        recovered = Box(60, 30, .72)
        wrong_box = Box(90, 40, .71)
        sheet = Sheet(0, [(30, 30), (60, 30)], [hit], [
            Box(32, 30, .95), recovered, wrong_box,
        ])
        base = baseline(sheet, .75)
        added, hits, wrong = second_pass(sheet, base, .70, "rotated")
        self.assertEqual(base.boxes, [hit])
        self.assertEqual(added, [recovered, wrong_box])
        self.assertEqual(hits, {1: recovered})
        self.assertEqual(wrong, [wrong_box])

    def test_zero_errors_and_zero_misses_have_undefined_shares(self):
        measures = {label: value for label, value, _ in metrics(
            np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        )}
        self.assertTrue(np.isnan(measures["Invisible misses / all baseline errors"]))
        self.assertTrue(np.isnan(measures["Rotation bank: invisible misses recovered"]))
        self.assertEqual(measures["Rotation bank: extra boxes/sheet"], 0)


if __name__ == "__main__":
    unittest.main()
