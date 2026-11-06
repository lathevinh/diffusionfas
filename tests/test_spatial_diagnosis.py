from __future__ import annotations

from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np

from diffusionfas.spatial_diagnosis import bottleneck, check_operating_point, crop_candidates, grouped_oracle


class SpatialDiagnosisTests(unittest.TestCase):
    def test_sklearn_auroc_roundoff_is_tolerated_but_decisions_are_exact(self):
        saved = {"FA": .1, "BFA": .7, "accepted_attack": 2, "accepted_bona_fide": 7,
                 "AUROC_usable": .9382465277777776}
        check_operating_point({**saved, "AUROC_usable": .9382465277777778}, saved)
        with self.assertRaises(ValueError):
            check_operating_point({**saved, "accepted_attack": 3}, saved)
        with self.assertRaises(ValueError):
            check_operating_point({**saved, "AUROC_usable": .9383}, saved)

    def test_grouped_ties_cannot_be_partially_accepted(self):
        scores = np.asarray([.1, .2, .2, .2, .3, .4])
        labels = np.asarray([0, 0, 1, 1, 0, 1])
        result = grouped_oracle(scores, labels, 3, 3, Fraction(1, 3))
        self.assertEqual(result["accepted_attack"], 0)
        self.assertEqual(result["accepted_bona_fide"], 1)
        self.assertEqual(result["threshold_diagnostic_only"], .2)
        self.assertEqual(result["max_BFA"], 1/3)
        self.assertFalse(result["deployable"])

    def test_original_failure_denominators_and_all_accept_endpoint(self):
        scores = np.asarray([.1, .2, .3])
        labels = np.asarray([0, 0, 1])
        result = grouped_oracle(scores, labels, 100, 4, Fraction(1, 100))
        self.assertEqual(result["attack_error_budget"], 1)
        self.assertEqual(result["FA"], .01)
        self.assertEqual(result["max_BFA"], .5)
        self.assertEqual(result["technical_failure_BFA_ceiling"], .5)
        self.assertGreater(result["threshold_diagnostic_only"], 1)
        self.assertEqual(bottleneck({"FA_within_constraint": False, "BFA": 1}, result, .85),
                         "technical_failure_ceiling")

    def test_independent_bruteforce_curve_on_random_tied_scores(self):
        rng = np.random.default_rng(123)
        for _ in range(40):
            scores = rng.integers(0, 6, 60)/5
            labels = rng.integers(0, 2, 60)
            for alpha in (Fraction(0), Fraction(1, 100), Fraction(1, 20), Fraction(1, 3)):
                attack, real = int(labels.sum())+3, int((1-labels).sum())+2
                result = grouped_oracle(scores, labels, attack, real, alpha)
                budget = attack*alpha.numerator//alpha.denominator
                points = []
                for threshold in [0., *np.unique(scores), np.nextafter(1., 2.)]:
                    accepted = scores < threshold
                    if int((accepted & (labels == 1)).sum()) <= budget:
                        points.append(int((accepted & (labels == 0)).sum()))
                self.assertEqual(result["accepted_bona_fide"], max(points))

    def test_invalid_scores_labels_and_counts_fail_explicitly(self):
        for scores, labels in ((np.array([np.nan]), np.array([0])),
                               (np.array([1.1]), np.array([0])),
                               (np.array([.2]), np.array([2]))):
            with self.assertRaises(ValueError):
                grouped_oracle(scores, labels, 1, 1, Fraction(1, 20))
        with self.assertRaises(ValueError):
            grouped_oracle(np.array([.1, .2]), np.array([1, 1]), 1, 1, Fraction(1, 20))

    def transaction(self):
        return {
            "selected_frame": {"width": 640, "height": 480},
            "geometry": {"crop_xyxy_half_open": [200, 100, 400, 400],
                         "alignment_matrix": [[1., 0., 0.], [0., 1., 0.]]},
            "crop_shape": [300, 200, 3],
            "landmarks": [[265, 200], [335, 200], [300, 240], [275, 280], [325, 280]],
        }

    def test_expanded_square_preserves_rotation_center_and_aspect(self):
        with patch("cv2.warpAffine", side_effect=AssertionError("no pixel processing")):
            result = crop_candidates(self.transaction())
        self.assertEqual(result["legacy_anisotropy_factor"], 1.5)
        square = result["candidates"]["expanded_legacy_square"]
        self.assertTrue(square["strict_pair_eligible"])
        self.assertEqual(square["side_source_pixels"], 300)
        matrix = np.asarray(square["source_to_RGB224_matrix"])
        self.assertTrue(np.allclose(matrix[:, :2], np.eye(2)*(224/300)))
        center = np.array([299.5, 249.5])
        self.assertTrue(np.allclose(matrix[:, :2]@center+matrix[:, 2], [111.5, 111.5]))
        quad = np.asarray(square["source_quad"])
        self.assertTrue(np.allclose(np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1), 299))

    def test_outside_square_never_shifted_shrunk_or_padded(self):
        transaction = self.transaction()
        transaction["selected_frame"]["width"] = 420
        transaction["geometry"]["crop_xyxy_half_open"] = [200, 100, 400, 400]
        result = crop_candidates(transaction)
        square = result["candidates"]["expanded_legacy_square"]
        self.assertFalse(square["new_crop_eligible"])
        self.assertFalse(square["strict_pair_eligible"])
        self.assertEqual(square["side_source_pixels"], 300)
        self.assertLess(square["minimum_source_margin_pixels"], 0)

    def test_legacy_border_support_and_invalid_geometry_are_not_silent(self):
        transaction = self.transaction()
        transaction["geometry"]["crop_xyxy_half_open"] = [200, 0, 400, 300]
        transaction["landmarks"] = (np.asarray(transaction["landmarks"])-[0, 100]).tolist()
        result = crop_candidates(transaction)
        self.assertFalse(result["legacy_real_pixel_support_conservative"])
        self.assertFalse(result["candidates"]["expanded_legacy_square"]["strict_pair_eligible"])
        transaction["geometry"]["alignment_matrix"][0][0] = 2
        with self.assertRaises(ValueError):
            crop_candidates(transaction)

    def test_bottleneck_is_goal_specific_not_certification(self):
        oracle = {"technical_failure_BFA_ceiling": 1., "max_BFA": .7}
        unsafe = {"FA_within_constraint": False, "BFA": .95}
        self.assertEqual(bottleneck(unsafe, oracle, .5),
                         "empirical_separation_available_policy_transfer_insufficient")
        self.assertEqual(bottleneck(unsafe, oracle, .85), "low_FA_separation_insufficient")
        self.assertEqual(bottleneck({"FA_within_constraint": True, "BFA": .6}, oracle, .5),
                         "current_policy_meets_this_empirical_goal")


if __name__ == "__main__":
    unittest.main()
