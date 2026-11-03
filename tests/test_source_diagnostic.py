from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from diffusionfas.cli import manifest_payload, run_prepare, run_profiles
from diffusionfas.data import Video
from diffusionfas.io import digest, read_json, write_json
from diffusionfas.metrics import Prediction, auroc_summary, paired_auroc_bootstrap
from diffusionfas.source_diagnostic import (
    AUTHORITY, BUDGET, checked_inputs, profile_map, run_diagnostic, validation_population,
)
from diffusionfas.source_preparation import source_media_entry
from diffusionfas.source_probe import stage_member

ROOT = Path(__file__).resolve().parents[1]


def video(domain="CASIA-FASD", label="bona_fide", partition="validation", key="sample"):
    return Video(key=f"{domain}/{key}", dataset=domain, group=key, subject=key,
                 partition=partition, label=label, timestamps=tuple(i / 16 for i in range(48)),
                 frame_hashes=(), media_sha256=None, decoded_sha256=None,
                 status="eligible_pending_tracking")


class SourceDiagnosticTests(unittest.TestCase):
    def test_population_includes_all_validation_classes_and_preserves_failures(self) -> None:
        rows = [video(), video(label="attack", key="attack"),
                replace(video(key="short"), timestamps=(0., 0.1)),
                video(partition="fitting", key="fit"),
                video(partition="calibration", key="calibration"),
                video(domain="OULU-NPU", partition="evaluation", key="target")]
        selected = validation_population(rows)
        self.assertEqual(len(selected), 3)
        self.assertEqual({v.label for v in selected}, {"bona_fide", "attack"})
        self.assertTrue(any(v.key.endswith("/short") for v in selected))

    def test_dedicated_opt_in_cannot_change_training_or_score_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            authority = dict(AUTHORITY)
            authority["higher_score_is_attack"] = False
            write_json(path, {"authority": authority, "budget": BUDGET})
            with self.assertRaises(PermissionError):
                checked_inputs(path, digest(path))
            second = Path(directory) / "training.json"
            write_json(second, {"authority": AUTHORITY, "budget": BUDGET,
                               "config": {"target": "OULU-NPU",
                                          "execution": {"source_training": True}}})
            with self.assertRaises(PermissionError):
                checked_inputs(second, digest(second))

    def test_existing_cli_validation_attack_guards_remain_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = read_json(ROOT / "configs/offline-v1.json")
            config["execution"].update(source_decode=True, source_scoring=True)
            write_json(root / "config.json", config)
            write_json(root / "manifest.json", manifest_payload(
                [video(), video(label="attack", key="attack")], {}))
            args = argparse.Namespace(config=root / "config.json", manifest=root / "manifest.json",
                                      out=root / "output.json", cache_root=root / "cache",
                                      include_attacks=True, partition="validation")
            with patch("diffusionfas.preprocessing.FasFaceProvider") as provider:
                with self.assertRaisesRegex(ValueError, "attack preparation restricted"):
                    run_prepare(args)
                provider.assert_not_called()
            with patch("diffusionfas.models.build_model") as model:
                with self.assertRaisesRegex(ValueError, "validation profile selection"):
                    run_profiles(args)
                model.assert_not_called()

    def test_reused_profiles_bind_exact_population_domains_and_published_means(self) -> None:
        bona = [video(domain=domain, key=f"bona-{i}") for i, domain in enumerate(
            ("CASIA-FASD", "MSU-MFSD", "SiW-Mv2"))]
        rows = [{"key": v.key, "dataset": v.dataset, "profile": [0.1] * 3, "loss": 0.1} for v in bona]
        expected = {"by_domain": {v.dataset: {"originals": 1, "mean_loss": 0.1} for v in bona},
                    "macro_loss": sum([0.1] * 3) / 3}
        self.assertEqual(len(profile_map({"records": rows}, bona, expected)), 3)
        with self.assertRaises(ValueError):
            profile_map({"records": [rows[0], rows[0], rows[2]]}, bona, expected)
        with self.assertRaises(ValueError):
            profile_map({"records": rows}, bona, {**expected, "macro_loss": 0.5})

    def test_shared_source_receipt_binding_rejects_attack_bonafide_label_mismatch(self) -> None:
        attack = video(label="attack")
        with self.assertRaisesRegex(ValueError, "identity/label"):
            source_media_entry(attack, {"dataset": attack.dataset, "video_id": "sample",
                                        "binary_label": "bona_fide"})

    def test_archive_attack_opt_in_is_validation_source_only_and_default_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            for domain in ("CASIA-FASD", "MSU-MFSD", "SiW-Mv2", "OULU-NPU"):
                for partition in ("fitting", "validation", "calibration", "evaluation"):
                    for opt_in in (False, True):
                        if domain != "OULU-NPU" and partition == "validation" and opt_in:
                            continue
                        with self.subTest(domain=domain, partition=partition, opt_in=opt_in):
                            entry = {"video": video(domain=domain, partition=partition,
                                                    label="attack").to_dict(),
                                     "member": "fixture.mp4"}
                            with self.assertRaisesRegex(ValueError, "staging restricted"):
                                stage_member(entry, root, root, root / "forbidden.mp4",
                                             stack, {}, time.monotonic() + 10,
                                             allow_validation_attacks=opt_in)
            self.assertFalse((root / "forbidden.mp4").exists())

    def test_diagnostic_worker_stages_accepted_attack_bytes_before_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "SiW").mkdir()
            payload = b"synthetic-original-for-staging-regression"
            with zipfile.ZipFile(root / "SiW/SiW-Mv2.zip", "w") as archive:
                archive.writestr("fixture.mp4", payload)
            (root / "scripts").mkdir()
            reader = root / "scripts/inventory_msu.py"
            reader.write_bytes(b"synthetic reader commitment")
            attack = replace(video(domain="SiW-Mv2", label="attack"),
                             media_sha256=hashlib.sha256(payload).hexdigest())
            config = read_json(ROOT / "configs/offline-v1.json")
            plan = {"config": config, "fas_root": str(root),
                    "msu_reader_sha256": digest(reader), "preprocessing_sha256": "fixture",
                    "detector": str(root / "unused.onnx"), "detector_sha256": "fixture",
                    "entries": [{"video": attack.to_dict(), "member": "fixture.mp4",
                                 "byte_size": len(payload), "metadata_failure": None}]}

            def stop_after_staging(staged_video, *args, **kwargs):
                self.assertEqual(staged_video.label, "attack")
                self.assertEqual(Path(staged_video.media_path).read_bytes(), payload)
                raise RuntimeError("intentional stop after verified staging")

            with patch("diffusionfas.source_diagnostic.checked_inputs", return_value=(plan, [])), \
                    patch("diffusionfas.source_diagnostic.FasFaceProvider"), \
                    patch("diffusionfas.source_diagnostic.decoder_policy",
                          return_value=({"SiW-Mv2": "opencv"}, None)), \
                    patch("diffusionfas.source_diagnostic.prepare_video",
                          side_effect=stop_after_staging) as prepare, \
                    patch("diffusionfas.source_diagnostic.configure_gpu") as gpu:
                with self.assertRaisesRegex(RuntimeError, "intentional stop after verified staging"):
                    run_diagnostic(root / "plan.json", "synthetic-plan")
                prepare.assert_called_once()
                gpu.assert_not_called()
            self.assertEqual(list((root / "staging").iterdir()), [])
            with patch("diffusionfas.source_diagnostic.checked_inputs", return_value=(plan, [])):
                with self.assertRaisesRegex(ValueError, "already attempted"):
                    run_diagnostic(root / "plan.json", "synthetic-plan")

    def test_auc_fixed_direction_ties_and_all_original_failure_bounds(self) -> None:
        rows = [Prediction("b", "source", "g1", "validation", "bona_fide", 0.5),
                Prediction("a", "source", "g1", "validation", "attack", 0.5),
                Prediction("failed", "source", "g2", "validation", "attack", None, "decode")]
        result = auroc_summary(rows)
        self.assertEqual(result["auroc_usable"], 0.5)
        self.assertEqual(result["all_original_missing_score_bounds"], [0.25, 0.75])
        self.assertEqual(result["coverage"]["attack"]["originals"], 2)
        reversed_rows = [replace(rows[0], score=1), replace(rows[1], score=0)]
        self.assertEqual(auroc_summary(reversed_rows)["auroc_usable"], 0)
        self.assertEqual(auroc_summary(reversed_rows)["score_direction"], "higher_is_attack")

    def test_paired_group_auc_bootstrap_uses_same_clusters_for_model_deltas(self) -> None:
        tied, perfect = [], []
        for group in range(3):
            for label in ("bona_fide", "attack"):
                row = Prediction(f"{group}/{label}", "source", str(group), "validation", label, 0.5)
                tied.append(row)
                perfect.append(replace(row, score=float(label == "attack")))
        result = paired_auroc_bootstrap({"frozen": tied, "adapter": perfect}, repetitions=100)
        self.assertEqual(result["declared_groups"], 3)
        self.assertEqual(result["valid_repetitions"], 100)
        self.assertEqual(result["intervals"]["adapter-minus-frozen"], [0.5, 0.5])
        with self.assertRaises(ValueError):
            paired_auroc_bootstrap({"frozen": tied, "adapter": list(reversed(perfect))})

    def test_missing_usable_class_is_explicit_not_a_fabricated_auc(self) -> None:
        rows = [Prediction("b", "source", "g1", "validation", "bona_fide", 0.5),
                Prediction("a", "source", "g2", "validation", "attack", None, "decode")]
        result = auroc_summary(rows)
        self.assertIsNone(result["auroc_usable"])
        self.assertEqual(result["unavailable_reason"], "insufficient_usable_classes")
        bootstrap = paired_auroc_bootstrap({"frozen": rows}, repetitions=20)
        self.assertEqual(bootstrap["invalid_usable_one_class"], 20)
        self.assertEqual(bootstrap["intervals"], {})
        self.assertEqual(bootstrap["unavailable_reason"], "insufficient_valid_bootstrap_replicates")
