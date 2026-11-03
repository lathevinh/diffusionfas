from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusionfas.data import addressed_seed
from diffusionfas.io import digest, write_json
from diffusionfas.mat_diagnostic import (
    AUTHORITY, BOOTSTRAP, BUDGET, SCORING, SCOPE, aggregate_profile, checked_inputs,
    score_frames, summarize,
)
import diffusionfas.mat_diagnostic as diagnostic
from diffusionfas.mat_input import POLICY, anatomical_masks


class MATDiagnosticTests(unittest.TestCase):
    def test_equal_region_frame_means_not_pooled_hole_pixels_or_maximum(self):
        profile = np.arange(64, dtype=np.float64).reshape(16, 4) / 100
        scores = aggregate_profile(profile)
        self.assertAlmostEqual(scores["primary"], .315)
        for i, name in enumerate(POLICY["mask_regions"]):
            self.assertAlmostEqual(scores[name], float(profile[:, i].mean()))
        for bad in (profile[:3], -profile, profile + np.nan, profile[:, :3]):
            with self.assertRaises(ValueError):
                aggregate_profile(bad)

    def test_cannot_enable_training_change_population_window_direction_or_caps(self):
        plan = {"scope": SCOPE, "budget": BUDGET, "authority": AUTHORITY,
                "scoring": SCORING, "policy": POLICY, "bootstrap": BOOTSTRAP}
        changes = [("authority", {**AUTHORITY, field: True}) for field in
                   ("training", "threshold_fitting", "target_pixels", "new_download", "automatic_retry")]
        changes += [("scoring", {**SCORING, "frames": 3}),
                    ("scoring", {**SCORING, "higher_score_is_attack": False}),
                    ("budget", {**BUDGET, "max_seconds": 3600})]
        for field, value in changes:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                write_json(path, {**plan, field: value})
                with self.assertRaises(PermissionError):
                    checked_inputs(path, digest(path))

    def test_full_T16_retains_preview_seeds_and_unquantized_hole_predictions(self):
        points = np.array([[192., 220.], [320., 220.], [256., 265.], [225., 308.], [287., 308.]])
        rgb = np.full((512, 512, 3), 127, dtype=np.uint8)
        geometry = {"output_landmarks": points.tolist()}
        samples = [{"clip_slot": i, "native_order": i, "landmarks": points.tolist()} for i in range(16)]
        video = SimpleNamespace(key="synthetic", frame_hashes=["native"] * 16)
        def fake_reconstruct(model, rgb, known, seed, device):
            prediction = rgb.astype(np.float32) / 127.5 - 1
            prediction[known == 0] = .1234567
            return prediction
        with patch("diffusionfas.mat_diagnostic.crop_native512", return_value=(rgb, geometry)), \
                patch("diffusionfas.mat_diagnostic.reconstruct", side_effect=fake_reconstruct) as forward:
            records, arrays = score_frames(None, video, samples, {i: rgb for i in range(16)}, torch.device("cpu"))
            self.assertEqual(forward.call_count, 64)
            self.assertEqual(arrays["crops"].shape, (16, 512, 512, 3))
            self.assertEqual(len(arrays), 65)
            for frame in records:
                self.assertEqual(frame["crop_sha256"], hashlib.sha256(rgb.tobytes()).hexdigest())
                for name, mask in anatomical_masks(points).items():
                    score = frame["masks"][name]
                    self.assertEqual(score["seed"], addressed_seed(20261103, video.key, frame["clip_slot"], name, "MAT-preview"))
                    self.assertEqual(score["hole_pixels"], int((mask == 0).sum()))
                    prediction = arrays[f"f{frame['clip_slot']:02d}-{name}"]
                    self.assertEqual(prediction.dtype, np.float32)
                    target = rgb[mask == 0].astype(np.float64) / 127.5 - 1
                    self.assertAlmostEqual(score["masked_mse"], float(((prediction.astype(np.float64) - target) ** 2).mean()), places=15)
            with self.assertRaisesRegex(ValueError, "T16"):
                score_frames(None, video, samples[:-1], {}, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "T16"):
                score_frames(None, video, list(reversed(samples)), {}, torch.device("cpu"))

    def test_round_trip_accepts_exact_entries_and_rejects_population_or_frame_drift(self):
        project = Path(diagnostic.__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "visual-approval.json", {"synthetic": True})
            write_json(root / "result.json", {"synthetic": True})
            baseline = {"entries": [{"video": {"key": "synthetic"}, "samples": list(range(16))}],
                        "fas_root": "source", "source_manifest": "manifest", "msu_reader_sha256": "reader"}
            preview = {"reuse_acquisition_root": "weights", "upstream_root": "upstream", "decoder_config": {}}
            plan = {"scope": SCOPE, "budget": BUDGET, "authority": AUTHORITY, "scoring": SCORING,
                    "policy": POLICY, "bootstrap": BOOTSTRAP, **baseline, "preview_root": str(root),
                    "baseline_root": str(root), "weights_root": "weights", "upstream_root": "upstream",
                    "decoder_config": {}, "acceptance_sha256": digest(root / "visual-approval.json"),
                    "baseline_result_sha256": digest(root / "result.json"),
                    "implementation_sha256": {str(p.relative_to(project)): digest(p)
                                              for directory in ("src", "scripts")
                                              for p in sorted((project / directory).rglob("*.py"))}}
            path = root / "plan.json"
            write_json(path, plan)
            with patch.object(diagnostic, "accepted_inputs", return_value=(preview, baseline)), \
                    patch.object(diagnostic, "acquired_weights"):
                self.assertEqual(checked_inputs(path, digest(path))["entries"], baseline["entries"])
                changed = root / "changed.json"
                write_json(changed, {**plan, "entries": [{"video": {"key": "synthetic"}, "samples": [0, 7, 15]}]})
                with self.assertRaisesRegex(ValueError, "population"):
                    checked_inputs(changed, digest(changed))

    def test_failed_attempt_is_not_reported_as_completed_or_given_fake_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "plan.json"
            write_json(path, {"entries": []})
            report = summarize(path, {"completed": False})
            self.assertFalse(report["completed"])
            self.assertIsNone(report["result"])
            self.assertIsNone(report["smoke"])

    def test_successful_receipt_rejects_pixel_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "plan.json"
            write_json(path, {"entries": []})
            pixels = root / "pixels/00.npz"
            pixels.parent.mkdir()
            pixels.write_bytes(b"synthetic immutable evidence")
            row = root / "records/result-00.json"
            write_json(row, {"failure": None, "pixels_sha256": digest(pixels)})
            write_json(root / "result.json", {"plan_sha256": digest(path), "records_sha256": {row.name: digest(row)}})
            self.assertTrue(summarize(path, {"completed": True})["completed"])
            pixels.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "pixel evidence"):
                summarize(path, {"completed": True})


if __name__ == "__main__":
    unittest.main()
