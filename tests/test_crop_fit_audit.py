from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diffusionfas.crop_fit_audit import audit_cache, population, quad_margin, window_counts
from diffusionfas.data import Video
from diffusionfas.io import digest, write_json

POINTS = np.array([[235., 160.], [285., 160.], [260., 195.], [240., 225.], [280., 225.]])


def video():
    return Video("CASIA-FASD/sample", "CASIA-FASD", "sample", "sample", "validation", "bona_fide",
                 tuple(0.1 + i / 8 for i in range(17)), (), None, None, "prepared")


class CropFitAuditTests(unittest.TestCase):
    def make_cache(self, root, points):
        row = video()
        path = root / "cache.npz"
        np.savez(path, timestamps=row.timestamps, native_timestamps=row.timestamps,
                 native_orders=np.arange(17), landmarks=points,
                 frames=np.array([{"must_not_decode": "image arrays"}], dtype=object))
        return replace(row, cache_path=str(path), cache_sha256=digest(path))

    def test_strict_real_pixel_quad_fit_and_dimension_validation(self):
        self.assertGreater(quad_margin(POINTS, 640, 480), 0)
        self.assertLess(quad_margin(POINTS, 200, 200), 0)
        self.assertLess(quad_margin(POINTS + [500, 0], 640, 480), 0)
        with self.assertRaises(ValueError):
            quad_margin(POINTS, 640., 480)

    def test_window_counts_require_all_sixteen_frames(self):
        valid = np.ones(17, dtype=np.bool_)
        self.assertEqual(window_counts(valid), (2, 2))
        valid[0] = False
        self.assertEqual(window_counts(valid), (1, 2))
        valid[8] = False
        self.assertEqual(window_counts(valid), (0, 2))
        with self.assertRaises(ValueError):
            window_counts(np.ones(15, dtype=np.bool_))

    def test_audit_never_decodes_image_arrays_or_video(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.make_cache(Path(directory), np.tile(POINTS, (17, 1, 1)))
            with patch("cv2.VideoCapture", side_effect=AssertionError("no video decode")), \
                    patch("cv2.warpAffine", side_effect=AssertionError("no pixel resampling")):
                result = audit_cache(row, [(640, 480)] * 17)
            self.assertTrue(result["all_cached_frames_fit"])
            self.assertTrue(result["middle_T16_fit"])
            self.assertEqual(result["fit_frames"], 17)
            self.assertEqual(result["image_arrays_decoded"], 0)

    def test_fixed_validation_window_cannot_be_replaced_by_a_better_window(self):
        with tempfile.TemporaryDirectory() as directory:
            points = np.tile(POINTS, (17, 1, 1))
            points[0] += [500, 0]
            row = self.make_cache(Path(directory), points)
            result = audit_cache(row, [(640, 480)] * 17)
            self.assertFalse(result["middle_T16_fit"])
            self.assertFalse(result["all_cached_frames_fit"])
            self.assertTrue(result["has_any_valid_T16_window"])
            self.assertEqual(result["valid_T16_windows"], 1)

    def test_training_full_grid_and_validation_middle_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            points = np.tile(POINTS, (17, 1, 1))
            points[-1] += [500, 0]
            row = self.make_cache(Path(directory), points)
            result = audit_cache(row, [(640, 480)] * 17)
            self.assertTrue(result["middle_T16_fit"])
            self.assertFalse(result["all_cached_frames_fit"])

    def test_invalid_landmarks_are_counted_not_treated_as_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            points = np.tile(POINTS, (17, 1, 1))
            points[7] = np.nan
            row = self.make_cache(Path(directory), points)
            result = audit_cache(row, [(640, 480)] * 17)
            self.assertEqual(result["geometry_failures"], {"celebahq_invalid_landmarks": 1})
            self.assertFalse(result["middle_T16_fit"])
            self.assertEqual(result["fit_frames"], 16)

    def test_native_dimensions_must_match_each_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.make_cache(Path(directory), np.tile(POINTS, (17, 1, 1)))
            dimensions = [(640, 480)] * 17
            dimensions[7] = (200, 200)
            result = audit_cache(row, dimensions)
            self.assertFalse(result["middle_T16_fit"])
            with self.assertRaises(ValueError):
                audit_cache(row, dimensions[:-1])

    def test_population_preserves_failures_and_excludes_other_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            val = video()
            fitting = replace(val, key="CASIA-FASD/fit", group="fit", subject="fit", partition="fitting")
            failure = replace(val, key="CASIA-FASD/failed", group="failed", subject="failed",
                              status="preparation_failure:missing_track_endpoint")
            attack = replace(val, key="CASIA-FASD/attack", label="attack")
            calibration = replace(val, key="CASIA-FASD/cal", group="cal", subject="cal", partition="calibration")
            write_json(root / "bona.json", {"schema_version": 1, "videos": [
                v.to_dict() for v in (val, fitting, failure, calibration)], "provenance": {}})
            write_json(root / "diag.json", {"schema_version": 1, "videos": [
                v.to_dict() for v in (val, failure, attack)], "provenance": {}})
            rows = population(root / "bona.json", root / "diag.json")
            self.assertEqual(len(rows), 4)
            self.assertTrue(any(v.status.startswith("preparation_failure:") for v in rows))
            self.assertTrue(any(v.label == "attack" for v in rows))
            self.assertFalse(any(v.partition == "calibration" for v in rows))
