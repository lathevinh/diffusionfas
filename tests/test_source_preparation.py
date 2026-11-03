from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diffusionfas.cli import manifest_payload
from diffusionfas.io import digest, read_json, write_json
from diffusionfas.source_preparation import build_plan, run_preparation, summarize_preparation
from test_native_replay import Fixture, fake_runtime


class PreparationFixture:
    def __init__(self, root: Path) -> None:
        self.fixture = Fixture(root)
        self.root, self.output = root, root / "batch"
        self.fas_root = self.fixture.source_root
        (self.fas_root / "src/fas").mkdir(parents=True)
        (self.fas_root / "src/fas/preprocessing.py").write_text("# fixture provider\n")
        (self.fas_root / "scripts/inventory_msu.py").write_text("# fixture archive reader\n")
        self.detector = root / "synthetic.onnx"
        self.detector.write_bytes(b"synthetic-detector-not-used")
        self.fixture.config["clip"]["duration"] = 2
        self.fixture.arguments()

    def plan(self) -> Path:
        return build_plan(
            self.root / "manifest.json", self.root / "source.json", self.root / "config.json",
            self.fas_root, self.detector, self.output, max_seconds=30,
            max_rss_bytes=8 * 1024**3, max_output_bytes=256 * 1024**2,
            source_manifest_sha=digest(self.root / "source.json"), detector_sha=digest(self.detector),
        )

    def stage(self, item, source_root, data_root, destination, stack, context, deadline):
        destination.write_bytes(self.fixture.media.read_bytes())


class SourcePreparationTests(unittest.TestCase):
    def test_complete_source_batch_verifies_windows_cleans_raw_and_preserves_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparationFixture(Path(directory))
            plan = fixture.plan()
            with fake_runtime(fixture.fixture), patch(
                "diffusionfas.source_preparation.stage_member", side_effect=fixture.stage,
            ):
                run_preparation(plan, digest(plan))
            report = summarize_preparation(plan, {"completed": True})
            self.assertTrue(report["completed"])
            self.assertEqual(report["original_denominator"], 1)
            self.assertEqual(report["prepared_originals"], 1)
            self.assertEqual(report["verified_cache_windows"], 3)
            self.assertEqual(report["successful_cache_grid_frames"], 24)
            self.assertEqual(report["target_video_decodes"], 0)
            rows = read_json(fixture.output / "prepared-manifest.json")["videos"]
            self.assertIsNone(rows[0]["media_path"])
            self.assertEqual(rows[0]["status"], "prepared")
            self.assertFalse(list((fixture.output / "staging").glob("source-*")))

    def test_membership_keeps_short_original_and_excludes_target_attack_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparationFixture(Path(directory))
            source = fixture.fixture.video
            short = replace(source, key="MSU-MFSD/short", group="short", subject="short",
                            timestamps=tuple(i / 16 for i in range(16)), frame_hashes=source.frame_hashes[:16],
                            media_sha256=None, decoded_sha256=None)
            target = replace(short, key="OULU-NPU/target", dataset="OULU-NPU", partition="evaluation",
                             group="target", subject="target")
            attack = replace(short, key="MSU-MFSD/attack", label="attack", group="attack", subject="attack")
            calibration = replace(short, key="MSU-MFSD/calibration", partition="calibration",
                                  group="calibration", subject="calibration")
            write_json(fixture.root / "population.json", manifest_payload(
                [source, short, target, attack, calibration], fixture.fixture.provenance))
            original_source = read_json(fixture.root / "source.json")
            original_source["rows"].append({"dataset": "MSU-MFSD", "video_id": "short", "binary_label": "bona_fide"})
            write_json(fixture.root / "population-source.json", original_source)
            payload = read_json(fixture.root / "population.json")
            payload["provenance"]["source_manifest_sha256"] = digest(fixture.root / "population-source.json")
            write_json(fixture.root / "bound-population.json", payload)
            plan = build_plan(
                fixture.root / "bound-population.json", fixture.root / "population-source.json",
                fixture.root / "config.json", fixture.fas_root, fixture.detector, fixture.output,
                max_seconds=30, max_rss_bytes=8 * 1024**3, max_output_bytes=256 * 1024**2,
                source_manifest_sha=digest(fixture.root / "population-source.json"),
                detector_sha=digest(fixture.detector),
            )
            planned = read_json(plan)
            self.assertEqual(len(planned["entries"]), 2)
            self.assertEqual(sum(e["metadata_failure"] == "insufficient_duration" for e in planned["entries"]), 1)
            self.assertFalse(planned["config"]["execution"]["source_training"])
            with fake_runtime(fixture.fixture), patch(
                "diffusionfas.source_preparation.stage_member", side_effect=fixture.stage,
            ) as stage:
                run_preparation(plan, digest(plan))
            self.assertEqual(stage.call_count, 1)
            rows = {r["key"]: r for r in read_json(fixture.output / "prepared-manifest.json")["videos"]}
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[short.key]["status"], "preparation_failure:insufficient_duration")
            for unchanged in (target, attack, calibration):
                self.assertEqual(rows[unchanged.key]["status"], unchanged.status)
            report = summarize_preparation(plan, {"completed": True})
            self.assertEqual(report["metadata_eligible_originals"], 1)
            self.assertEqual(report["failures_by_reason"]["insufficient_duration"], 1)

    def test_partial_run_accounts_for_interrupted_original_and_cleans_owned_raw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparationFixture(Path(directory))
            plan = fixture.plan()

            def failing_stage(*args):
                fixture.stage(*args)
                raise ValueError("synthetic staged-byte mismatch")

            with fake_runtime(fixture.fixture), patch(
                "diffusionfas.source_preparation.stage_member", side_effect=failing_stage,
            ), self.assertRaisesRegex(ValueError, "staged-byte mismatch"):
                run_preparation(plan, digest(plan))
            report = summarize_preparation(plan, {"completed": False})
            self.assertFalse(report["completed"])
            self.assertEqual(report["counts_by_domain_partition"]["MSU-MFSD:fitting"]["interrupted"], 1)
            self.assertFalse(list((fixture.output / "staging").glob("source-*")))
            self.assertFalse((fixture.output / "completion.json").exists())

    def test_plan_and_metadata_commitments_prevent_retry_or_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparationFixture(Path(directory))
            plan = fixture.plan()
            with self.assertRaises(ValueError):
                fixture.plan()
            with self.assertRaises(ValueError):
                run_preparation(plan, "0" * 64)
            with (fixture.root / "manifest.json").open("a") as stream:
                stream.write(" ")
            with patch("diffusionfas.source_preparation.stage_member") as stage, self.assertRaisesRegex(
                ValueError, "metadata changed",
            ):
                run_preparation(plan, digest(plan))
            stage.assert_not_called()

    def test_small_output_budget_fails_before_batch_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparationFixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "exceeds approved output cap"):
                build_plan(
                    fixture.root / "manifest.json", fixture.root / "source.json",
                    fixture.root / "config.json", fixture.fas_root, fixture.detector, fixture.output,
                    max_seconds=30, max_rss_bytes=8 * 1024**3, max_output_bytes=1,
                    source_manifest_sha=digest(fixture.root / "source.json"),
                    detector_sha=digest(fixture.detector),
                )
            self.assertFalse(fixture.output.exists())
