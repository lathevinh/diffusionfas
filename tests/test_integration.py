from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from diffusionfas.baselines import FeatureGaussian, clip_feature
from diffusionfas.cli import run_calibrate, run_evaluate
from diffusionfas.fas_import import import_metadata
from diffusionfas.io import digest, read_json, write_json
from diffusionfas.models import build_model
from diffusionfas.scoring import residual_profile

from test_contracts import video
from test_models import TINY


class ImportTests(unittest.TestCase):
    def test_signed_metadata_preserves_missing_transaction_failure_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "transactions").mkdir()
            (root / "media").mkdir()
            receipt = {
                "video_id": "ok", "media_sha256": "media", "decode_status": "decoded",
                "decoded_rgb_sequence_sha256": "decoded",
                "frame_index": [{"timestamp_seconds": str(i / 30), "rgb_sha256": str(i)} for i in range(90)],
            }
            write_json(root / "media/0.json", receipt)
            transaction = {"video_id": "ok", "binary_label": "bona_fide",
                           "media_sha256": "media", "media_record_sha256": digest(root / "media/0.json")}
            write_json(root / "transactions/0.json", transaction)
            rows = [
                {"dataset": "CASIA-FASD", "video_id": "ok", "official_split": "train",
                 "record_path": str(root / "transactions/0.json"),
                 "record_sha256": digest(root / "transactions/0.json"),
                 "binary_label": "bona_fide", "group_id": "ok", "subject_id": "ok"},
                {"dataset": "OULU-NPU", "video_id": "failed", "official_split": "train",
                 "record_path": None, "record_sha256": None, "detector_status": "not_run",
                 "failure_kind": "pre_detector", "binary_label": "attack",
                 "group_id": "failed", "subject_id": "failed"},
                {"dataset": "SiW-Mv2", "video_id": "excluded", "official_split": "train",
                 "source_use_eligible": False},
            ]
            write_json(root / "manifest.json", {"rows": rows, "source_use_binding": {"policy_id": "fixture"}})
            manifest_sha = digest(root / "manifest.json")
            write_json(root / "partitions.json", {
                "source_manifest_sha256": manifest_sha,
                "contexts": [{"fitting": ["CASIA-FASD/ok", "OULU-NPU/failed"],
                              "validation": [], "calibration": [], "evaluation": []}],
            })
            imported, provenance = import_metadata(
                root / "manifest.json", root / "partitions.json", root,
                manifest_sha, digest(root / "partitions.json"),
            )
            self.assertEqual(len(imported), 2)
            self.assertEqual(imported[1].status, "historical_pre_detector_failure")
            self.assertEqual(imported[1].timestamps, ())
            self.assertEqual(provenance["excluded_source_originals"], 1)
            self.assertEqual(provenance["new_video_decodes"], 0)


class BaselineTests(unittest.TestCase):
    def test_fixed_noise_baseline_runs_and_rejects_other_levels(self) -> None:
        config = {"model": {"variant": "scratch_fixed_noise"}, "scratch": TINY,
                  "scoring": {"alpha_bars": [0.5]}}
        model = build_model(config)
        clips = torch.randn(1, 2, 3, 16, 16)
        result = residual_profile(model, clips, ["fixture"], alpha_bars=[0.5], draws=1, seed=1)
        self.assertEqual(result.shape, (1, 1))
        with self.assertRaises(ValueError):
            residual_profile(model, clips, ["fixture"], alpha_bars=[0.9], draws=1, seed=1)

    def test_temporal_features_and_source_only_shrinkage_gaussian(self) -> None:
        frames = np.arange(24, dtype=float).reshape(4, 6)
        self.assertEqual(clip_feature(frames, temporal=False).shape, (6,))
        self.assertEqual(clip_feature(frames, temporal=True).shape, (18,))
        videos = [video(key=str(i), group=str(i), subject=str(i)) for i in range(8)]
        features = np.random.default_rng(42).normal(size=(8, 6))
        model = FeatureGaussian.fit(features, videos, "OULU-NPU")
        self.assertTrue(np.isfinite(model.score(features)).all())
        with self.assertRaises(ValueError):
            FeatureGaussian.fit(features, [replace(v, dataset="OULU-NPU") for v in videos], "OULU-NPU")


class CalibrationIntegrationTests(unittest.TestCase):
    def test_source_calibration_to_locked_target_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = {
                "target": "OULU-NPU", "scoring_binding": "synthetic-only",
                "records": [
                    {"key": f"v{i}", "dataset": "CASIA-FASD", "group": f"v{i}",
                     "partition": "validation", "label": "bona_fide", "profile": [float(i)],
                     "failure": None} for i in range(8)
                ],
            }
            calibration = {
                "target": "OULU-NPU", "scoring_binding": "synthetic-only",
                "records": [
                    {"key": f"c{i}", "dataset": "CASIA-FASD", "group": f"c{i}",
                     "partition": "calibration", "label": "bona_fide", "profile": [float(i)],
                     "failure": None} for i in range(20)
                ],
            }
            write_json(root / "validation.json", validation)
            write_json(root / "calibration.json", calibration)
            run_calibrate(argparse.Namespace(validation=root / "validation.json",
                                             calibration=root / "calibration.json",
                                             out=root / "policy.json", regime="strict", rate=0.05))
            profiles = {
                "target": "OULU-NPU", "scoring_binding": "synthetic-only",
                "config": {"execution": {"target_evaluation": True}},
                "records": [
                    {"key": "b", "dataset": "OULU-NPU", "group": "g1", "partition": "evaluation",
                     "label": "bona_fide", "profile": [0.], "failure": None},
                    {"key": "a", "dataset": "OULU-NPU", "group": "g2", "partition": "evaluation",
                     "label": "attack", "profile": [40.], "failure": None},
                ],
            }
            write_json(root / "profiles.json", profiles)
            run_evaluate(argparse.Namespace(profiles=root / "profiles.json",
                                            calibration=root / "policy.json", out=root / "result.json", bootstrap=50))
            report = read_json(root / "result.json")
            self.assertEqual(report["attack"]["accepted"], 0)
            self.assertEqual(report["bonafide"]["accepted"], 1)
            self.assertEqual(report["auroc_usable"], 1)
            profiles["scoring_binding"] = "different-model"
            write_json(root / "wrong.json", profiles)
            with self.assertRaises(ValueError):
                run_evaluate(argparse.Namespace(profiles=root / "wrong.json",
                                                calibration=root / "policy.json", out=root / "bad.json", bootstrap=50))


if __name__ == "__main__":
    unittest.main()
