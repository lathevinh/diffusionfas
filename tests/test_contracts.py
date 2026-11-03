from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diffusionfas.cli import require_permission
from diffusionfas.data import DOMAINS, Video, clip_start, validate_isolation, window_indices
from diffusionfas.io import digest, read_json, require_private_output, write_json
from diffusionfas.metrics import Prediction, attack_threshold, operating_metrics, oracle_bfa, paired_bootstrap, strict_threshold
from diffusionfas.preflight import inventory
from diffusionfas.preprocessing import MediaFailure, interpolate_track
from diffusionfas.scoring import RobustNormalizer


def video(**changes: object) -> Video:
    values = dict(key="video-1", dataset="CASIA-FASD", group="subject-1", subject="subject-1",
                  partition="fitting", label="bona_fide",
                  timestamps=tuple(i / 30 for i in range(90)), frame_hashes=(),
                  media_sha256=None, decoded_sha256=None, status="eligible_pending_tracking")
    values.update(changes)
    return Video(**values)


class SamplingTests(unittest.TestCase):
    def test_sixteen_unique_frames_and_fixed_midpoint(self) -> None:
        row = video()
        start = clip_start(row, 2)
        self.assertAlmostEqual(start, 0.5)
        selected = window_indices(row.timestamps, start)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len(set(selected)), 16)

    def test_addressed_training_windows_change(self) -> None:
        row = video()
        self.assertEqual(clip_start(row, 2, 1), clip_start(row, 2, 1))
        self.assertNotEqual(clip_start(row, 2, 1), clip_start(row, 2, 2))

    def test_short_video_is_not_padded(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient_duration"):
            clip_start(video(timestamps=(0.0, 0.1)), 2)

    def test_low_fps_cannot_duplicate_frames(self) -> None:
        with self.assertRaises(ValueError):
            window_indices((0.0, 0.5, 1.0, 1.5, 2.0), 0)

    def test_invalid_times_rejected(self) -> None:
        for timestamps in ((0, 0), (0, float("nan")), (1, 0)):
            with self.assertRaises(ValueError):
                video(timestamps=timestamps)
        with self.assertRaises(ValueError):
            window_indices(video().timestamps, float("inf"))

    def test_isolation_uses_subject_and_full_content(self) -> None:
        row = video()
        for other in (
            replace(row, key="other", partition="validation"),
            replace(row, key="other", dataset="MSU-MFSD", group="other", subject="other",
                    media_sha256="same", partition="validation"),
        ):
            first = replace(row, media_sha256="same")
            with self.assertRaises(ValueError):
                validate_isolation([first, other])

    def test_inventory_preserves_failures(self) -> None:
        result = inventory([video(), video(key="failed", group="failed", subject="failed",
                                          timestamps=(), status="historical_decode_failure")])
        bucket = result["populations"]["CASIA-FASD/fitting/bona_fide"]
        self.assertEqual(bucket["originals"], 2)
        self.assertEqual(bucket["eligible_by_seconds_before_new_tracking"]["2"], 1)
        self.assertEqual(result["new_media_decodes"], 0)


class IOAndTrackingTests(unittest.TestCase):
    def test_immutable_json_and_hash_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_json(path, {"a": 1})
            self.assertEqual(read_json(path, digest(path)), {"a": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {})
            with self.assertRaises(ValueError):
                read_json(path, "wrong")
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "nan.json", {"a": float("nan")})

    def test_private_output_cannot_be_in_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            with self.assertRaises(ValueError):
                require_private_output(root / "private/rows.json")

    def test_all_execution_gates_closed(self) -> None:
        config = {"execution": {name: False for name in (
            "source_decode", "source_training", "source_scoring", "target_evaluation",
        )}}
        for name in config["execution"]:
            with self.assertRaises(PermissionError):
                require_permission(config, name)

    def test_track_interpolates_bounded_interior_gap(self) -> None:
        detection = (np.array([0., 0., 20., 20.]), np.ones((5, 2)))
        boxes, points = interpolate_track([detection, None, None, detection])
        self.assertTrue(np.isfinite(boxes).all())
        self.assertEqual(points.shape, (4, 5, 2))
        for detections in ([None, detection], [detection, None],
                           [detection, None, None, None, detection]):
            with self.assertRaises(MediaFailure):
                interpolate_track(detections)

    def test_track_jump_is_failure_not_anomaly(self) -> None:
        first = (np.array([0., 0., 20., 20.]), np.ones((5, 2)))
        second = (np.array([100., 100., 120., 120.]), np.ones((5, 2)))
        with self.assertRaisesRegex(MediaFailure, "jump"):
            interpolate_track([first, second])


class MetricTests(unittest.TestCase):
    def calibration(self, label: str, scores: list[float]) -> list[Prediction]:
        return [Prediction(str(i), "CASIA-FASD", str(i), "calibration", label, score)
                for i, score in enumerate(scores)]

    def test_attack_threshold_integer_budget_and_ties(self) -> None:
        rows = self.calibration("attack", list(map(float, range(20))))
        threshold = attack_threshold(rows, "OULU-NPU", 0.05)
        self.assertEqual(threshold, 1)
        self.assertEqual(sum(r.score < threshold for r in rows), 1)
        ties = self.calibration("attack", [0.0] * 20)
        self.assertEqual(attack_threshold(ties, "OULU-NPU"), 0)

    def test_strict_threshold_includes_boundary_bonafide(self) -> None:
        rows = self.calibration("bona_fide", list(map(float, range(20))))
        threshold = strict_threshold(rows, "OULU-NPU", 0.05)
        self.assertEqual(sum(r.score < threshold for r in rows), 19)
        with self.assertRaises(ValueError):
            strict_threshold(rows[:3], "OULU-NPU")
        with self.assertRaises(ValueError):
            strict_threshold(self.calibration("attack", [0.] * 20), "OULU-NPU")

    def test_failures_do_not_enlarge_attack_budget(self) -> None:
        rows = self.calibration("attack", list(map(float, range(19))))
        rows.append(Prediction("failed", "CASIA-FASD", "failed", "calibration",
                               "attack", None, "decode"))
        self.assertEqual(attack_threshold(rows, "OULU-NPU"), 0)

    def test_failure_denominator_and_auroc(self) -> None:
        rows = [
            Prediction("a", "OULU-NPU", "s1", "evaluation", "attack", 2),
            Prediction("b", "OULU-NPU", "s1", "evaluation", "bona_fide", 0),
            Prediction("c", "OULU-NPU", "s2", "evaluation", "bona_fide", None, "decode"),
        ]
        result = operating_metrics(rows, 1)
        self.assertEqual(result["auroc_usable"], 1)
        self.assertEqual(result["bonafide"]["acceptance_all_originals"], 0.5)
        self.assertEqual(result["bonafide"]["acceptance_usable"], 1)
        with self.assertRaises(ValueError):
            Prediction("f", "OULU-NPU", "s", "evaluation", "attack", 0, "failure")

    def test_target_calibration_forbidden(self) -> None:
        with self.assertRaises(ValueError):
            attack_threshold(self.calibration("attack", [1.] * 20), "CASIA-FASD")

    def test_grouped_paired_bootstrap(self) -> None:
        rows = []
        for group in range(4):
            for label, score in (("attack", 2.), ("bona_fide", 0.)):
                rows.append(Prediction(f"{group}/{label}", "OULU-NPU", str(group),
                                       "evaluation", label, score))
        result = paired_bootstrap({"a": rows, "b": rows}, {"a": 1, "b": 1}, repetitions=50)
        self.assertEqual(result["invalid_one_class"], 0)
        self.assertEqual(result["intervals"]["b-minus-a"], [[0., 0.], [0., 0.]])

    def test_oracle_matches_brute_force_with_ties_and_failures(self) -> None:
        rows = [
            Prediction("a1", "OULU-NPU", "s1", "evaluation", "attack", 1.),
            Prediction("a2", "OULU-NPU", "s2", "evaluation", "attack", None, "decode"),
            Prediction("b1", "OULU-NPU", "s1", "evaluation", "bona_fide", 0.),
            Prediction("b2", "OULU-NPU", "s2", "evaluation", "bona_fide", 1.),
            Prediction("b3", "OULU-NPU", "s3", "evaluation", "bona_fide", None, "decode"),
        ]
        best = 0.
        for threshold in (-1., 0., 0.5, 1., 1.5, 2.):
            metric = operating_metrics(rows, threshold)
            if metric["attack"]["accepted"] <= 0:
                best = max(best, metric["bonafide"]["acceptance_all_originals"])
        self.assertEqual(oracle_bfa(rows)["BFA_all_originals"], best)
        self.assertTrue(oracle_bfa(rows)["diagnostic_only"])

    def test_normalizer_source_validation_only(self) -> None:
        rows = [dict(dataset="CASIA-FASD", partition="validation", label="bona_fide") for _ in range(8)]
        profiles = np.arange(24, dtype=float).reshape(8, 3)
        normalizer = RobustNormalizer.fit(profiles, rows, "OULU-NPU")
        self.assertEqual(normalizer.transform(profiles).shape, (8,))
        with self.assertRaises(ValueError):
            RobustNormalizer.fit(np.ones((8, 3)), rows, "OULU-NPU")
        with self.assertRaises(ValueError):
            RobustNormalizer.fit(profiles, rows, "CASIA-FASD")


if __name__ == "__main__":
    unittest.main()
