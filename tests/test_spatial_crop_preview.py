import hashlib
import unittest

import cv2
import numpy as np

from diffusionfas.spatial_crop_preview import preview_keys, render_pair, require_bicubic_support
import test_spatial_diagnosis as fixtures


class SpatialCropPreviewTests(unittest.TestCase):
    def test_rotation_mask_roundoff_does_not_replace_geometric_support_check(self):
        rotation = cv2.getRotationMatrix2D((300, 250), 4.2, 1.)
        mask = cv2.warpAffine(np.ones((480, 640), np.float32), rotation, (640, 480),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        roi = mask[100:400, 200:400]
        self.assertTrue(np.any(roi != 1))
        self.assertLess(np.max(np.abs(roi-1)), 2e-7)
        require_bicubic_support(rotation, (200, 100, 400, 400), 640, 480)
        with self.assertRaisesRegex(ValueError, "footprint"):
            require_bicubic_support(np.array([[1., 0, 0], [0, 1., 0]]), (0, 0, 100, 100), 640, 480)

    def test_synthetic_native_render_matches_legacy_and_keeps_square_aspect(self):
        transaction = fixtures.SpatialDiagnosisTests().transaction()
        y, x = np.indices((480, 640))
        rgb = np.stack((x % 256, y % 256, (x+y) % 256), axis=-1).astype(np.uint8)
        transaction["selected_frame"]["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        old, new, record = render_pair(rgb, transaction)
        expected = cv2.resize(rgb[100:400, 200:400], (224, 224), interpolation=cv2.INTER_CUBIC)
        self.assertTrue(np.array_equal(old, expected))
        self.assertEqual(new.shape, (224, 224, 3))
        self.assertEqual(record["padding_fraction"], 0)
        self.assertTrue(record["actual_square_interpolation_support_verified"])
        matrix = np.asarray(record["candidates"]["expanded_legacy_square"]["source_to_RGB224_matrix"])
        self.assertAlmostEqual(np.linalg.norm(matrix[:, 0]), np.linalg.norm(matrix[:, 1]))

    def test_changed_native_frame_is_rejected(self):
        transaction = fixtures.SpatialDiagnosisTests().transaction()
        transaction["selected_frame"]["rgb_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "exact historical"):
            render_pair(np.zeros((480, 640, 3), np.uint8), transaction)

    def test_score_independent_selection_is_exact_balanced_and_order_invariant(self):
        records, contexts = [], []
        for target in ("a", "b", "c", "d"):
            keys = []
            for label in ("bona_fide", "attack"):
                for i in range(4):
                    key = f"{target}/{label}/{i}"
                    keys.append(key)
                    records.append({"key": key, "label": label, "geometry": {
                        "candidates": {"expanded_legacy_square": {"strict_pair_eligible": i != 0}}}})
            contexts.append({"target": target, "evaluation": keys})
        first = preview_keys({"contexts": contexts}, records)
        self.assertEqual(len(first), 16)
        self.assertEqual(first, preview_keys({"contexts": contexts}, list(reversed(records))))
        for record in records:
            record["model_score"] = float("nan")
        self.assertEqual(first, preview_keys({"contexts": contexts}, records))
        with self.assertRaisesRegex(ValueError, "unique"):
            preview_keys({"contexts": contexts}, records+[records[0]])


if __name__ == "__main__":
    unittest.main()
