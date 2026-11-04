from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from diffusionfas.dino_comparison import (
    AUTHORITY, BUDGET, BRANCHES, POLICY, SEEDS, checked_inputs, frame_scores,
    pairwise_complementarity, strict_crop, summarize, validate_models, validate_role_isolation,
)
from diffusionfas.io import digest, write_json
from diffusionfas.metrics import Prediction
from diffusionfas.preprocessing import MediaFailure


class DINOComparisonTests(unittest.TestCase):
    def test_isolation_retains_source_failures_without_fabricating_frame_evidence(self):
        context = {"fitting": ["good", "failure"], "validation": [], "calibration": []}
        good = {"dataset": "CASIA-FASD", "detector_status": "success", "selected_frame": {"rgb_sha256": "source"}}
        failure = {"dataset": "CASIA-FASD", "detector_status": "failed", "selected_frame": None}
        validate_role_isolation(context, {"good": good, "failure": failure}, {"queried"})
        with self.assertRaisesRegex(ValueError, "selected-frame"):
            validate_role_isolation(context, {"good": {**good, "selected_frame": None}, "failure": failure}, set())
        with self.assertRaisesRegex(PermissionError, "dataset"):
            validate_role_isolation(context, {"good": good, "failure": {**failure, "dataset": "MSU-MFSD"}}, set())
        with self.assertRaisesRegex(PermissionError, "exact frame"):
            validate_role_isolation(context, {"good": good, "failure": failure}, {"source"})

    def test_exact_historical_resize_with_strict_native_cubic_support(self):
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        def preprocessing(bounds):
            x1, y1, x2, y2 = bounds
            def crop(rgb, box, landmarks):
                return rgb[y1:y2, x1:x2], {"crop_xyxy_half_open": bounds,
                                           "alignment_matrix": [[1., 0., 0.], [0., 1., 0.]]}
            return SimpleNamespace(shared_context_crop=crop)
        sample = {"box": [20, 20, 60, 60], "landmarks": np.zeros((5, 2))}
        result, evidence = strict_crop(rgb, sample, preprocessing([12, 16, 66, 80]))
        np.testing.assert_array_equal(result, cv2.resize(rgb[16:80, 12:66], (518, 518), interpolation=cv2.INTER_CUBIC))
        self.assertEqual(evidence["rectangle_width_height"], [54, 64])
        self.assertTrue(evidence["cubic_support_real_pixels"])
        for bounds in ([0, 16, 66, 80], [12, 16, 100, 80], [12, 0, 66, 80], [12, 16, 66, 100]):
            with self.assertRaisesRegex(MediaFailure, "cubic_footprint"):
                strict_crop(rgb, sample, preprocessing(bounds))

    def models(self):
        return {str(seed): {
            "heads": {b: {"seed": seed, "coefficients": [1.] + [0.] * 1535 + [i * .3]}
                      for i, b in enumerate(BRANCHES)},
            "calibrators": {b: {"slope": 1 + i, "intercept": -.2}
                            for i, b in enumerate(BRANCHES)}} for seed in SEEDS}

    def test_all_seeds_use_existing_head_calibration_and_probability_average(self):
        models = self.models()
        features = {b: np.zeros((16, 1536), dtype=np.float32) for b in BRANCHES}
        features["dino_reg"][:, 0] = np.arange(16) / 10
        features["dino_plain"][:, 0] = np.arange(16)[::-1] / 10
        fitting = SimpleNamespace(
            head_logits=lambda m, x: x @ np.asarray(m["coefficients"][:-1]) + m["coefficients"][-1],
            calibrate=lambda m, x: 1 / (1 + np.exp(-(m["slope"] * x + m["intercept"]))))
        scores = frame_scores(features, models, fitting)
        self.assertEqual(set(scores), set(map(str, SEEDS)))
        for seed in SEEDS:
            p = scores[str(seed)]
            expected = (np.asarray(p["dino_reg"]) + np.asarray(p["dino_plain"])) / 2
            np.testing.assert_array_equal(p["two_dino"], expected)
        for name in BRANCHES:
            bad = {**features, name: features[name][:3]}
            with self.assertRaisesRegex(ValueError, "T16"):
                frame_scores(bad, models, fitting)
        model = models[str(SEEDS[0])]
        model["heads"]["dino_reg"]["seed"] = SEEDS[1]
        with self.assertRaisesRegex(ValueError, "head"):
            validate_models(model, SEEDS[0])

    def rows(self, scores):
        return [Prediction(f"v{i}", "MSU-MFSD", "group", "validation",
                           "attack" if i < 2 else "bona_fide", s) for i, s in enumerate(scores)]

    def test_complementarity_is_pair_ordering_not_threshold_rescue(self):
        dino = self.rows([.2, .9, .3, .8])
        mat = self.rows([.6, .7, .4, .65])
        table = pairwise_complementarity(dino, mat)
        self.assertEqual(table["counts"], [[1, 0, 1], [0, 0, 0], [0, 0, 2]])
        self.assertEqual(table["attack_bonafide_pairs"], 4)
        self.assertEqual(table["DINO_wrong_MAT_correct"], 1)
        self.assertTrue(table["not_operating_point_rescue"])
        self.assertTrue(table["dependent_pairs_not_independent_trials"])
        reverse = pairwise_complementarity(mat, dino)
        self.assertEqual(reverse["DINO_correct_MAT_wrong"], 1)

    def test_ties_remain_explicit_and_identity_order_is_mandatory(self):
        dino = self.rows([.5, .5, .5, .5])
        mat = self.rows([.9, .9, .1, .1])
        table = pairwise_complementarity(dino, mat)
        self.assertEqual(table["counts"], [[0, 0, 0], [0, 0, 4], [0, 0, 0]])
        self.assertEqual(table["DINO_wrong_MAT_correct"], 0)
        with self.assertRaisesRegex(ValueError, "ordered"):
            pairwise_complementarity(dino, list(reversed(mat)))

    def test_no_authority_for_seed_selection_fitting_or_new_pixels(self):
        base = {"scope": "MSU_matched_video_frozen_DINO_MAT_development_diagnostic",
                "authority": AUTHORITY, "budget": BUDGET, "policy": POLICY, "seeds": list(SEEDS)}
        changes = [("authority", {**AUTHORITY, field: True}) for field in
                   ("new_training", "threshold_selection", "fusion_fitting", "new_download",
                    "new_calibration_or_target_pixels", "automatic_retry")]
        changes += [("seeds", [SEEDS[0]]), ("policy", {**POLICY, "same_crop_as_MAT": True})]
        for field, value in changes:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                write_json(path, {**base, field: value})
                with self.assertRaises(PermissionError):
                    checked_inputs(path, digest(path))

    def test_failure_does_not_fabricate_result_and_success_checks_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "plan.json"
            write_json(path, {"entries": []})
            self.assertFalse(summarize(path, {"completed": False})["completed"])
            feature = root / "features/00.npy"
            feature.parent.mkdir()
            feature.write_bytes(b"synthetic evidence")
            write_json(root / "result.json", {"plan_sha256": digest(path),
                                              "files_sha256": {"features/00.npy": digest(feature)}})
            self.assertTrue(summarize(path, {"completed": True})["completed"])
            feature.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "evidence drift"):
                summarize(path, {"completed": True})


if __name__ == "__main__":
    unittest.main()
