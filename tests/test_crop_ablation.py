from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from diffusers import DDPMScheduler
import test_crop_preview as preview_tests

from diffusionfas.celebahq_geometry import REAL_PIXEL_POLICY
from diffusionfas.crop_ablation import (
    ARMS, AUTHORITY, BUDGET, SCORING, SEED, checked_inputs, compare_predictions,
    full_middle_samples, render_pair, run_ablation, score_pair, summarize, validate_population,
)
from diffusionfas.data import addressed_seed
from diffusionfas.fas_import import MANIFEST_SHA
from diffusionfas.dataset import ClipCache
from diffusionfas.io import digest, read_json, write_json
from diffusionfas.metrics import Prediction
from diffusionfas.preprocessing import MediaFailure
from test_crop_preview import POINTS, video


class ZeroDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scheduler = DDPMScheduler(num_train_timesteps=16, prediction_type="epsilon")
        self.backbone = torch.nn.Linear(1, 1).requires_grad_(False)
        self.adapter = None

    def cuda(self):
        return self

    def forward(self, noisy, timestep):
        return torch.zeros_like(noisy)


class CropAblationTests(unittest.TestCase):
    def test_population_requires_exact_validation_subset_without_fitting_or_target(self):
        rows = [video("MSU-MFSD", "bona_fide" if i < 6 else "attack", f"v{i}") for i in range(22)]
        validate_population(rows)
        for bad in (rows[:-1], [*rows[:-1], rows[0]],
                    [replace(rows[0], partition="fitting"), *rows[1:]],
                    [replace(rows[0], dataset="OULU-NPU"), *rows[1:]]):
            with self.assertRaises(PermissionError):
                validate_population(bad)

    def test_ablation_cannot_enable_training_or_change_score_recipe(self):
        for authority, scoring in (({**AUTHORITY, "training": True}, SCORING),
                                   (AUTHORITY, {"alpha_bars": [.5], "draws": 2})):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                write_json(path, {"scope": "MSU_frozen_matched_crop_development_diagnostic",
                                  "authority": authority, "budget": BUDGET,
                                  "policy": REAL_PIXEL_POLICY,
                                  "config": {"seed": SEED, "model": {"variant": "pretrained_spatial"},
                                             "scoring": scoring}})
                with self.assertRaises(PermissionError):
                    checked_inputs(path, digest(path))

    def test_checked_plan_accepts_json_round_trip_but_rejects_changed_video_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [video("MSU-MFSD", "bona_fide" if i < 6 else "attack", f"v{i}") for i in range(22)]
            config = {"seed": SEED, "scoring": SCORING, "model": {"variant": "pretrained_temporal"},
                      "execution": {"source_training": True, "source_decode": False, "source_scoring": False}}
            diagnostic = root / "diagnostic"
            diagnostic.mkdir()
            write_json(diagnostic / "plan.json", {"config": config})
            write_json(root / "visual-approval.json", {"synthetic": True})
            source = root / "source.json"
            write_json(source, {"rows": [{"dataset": "MSU-MFSD", "video_id": f"v{i}"} for i in range(22)]})
            preview = {"diagnostic_root": str(diagnostic), "diagnostic_plan_sha256": digest(diagnostic / "plan.json"),
                       "real_pixel_cohort": {}, "fas_root": "unused", "source_manifest": str(source),
                       "preprocessing_sha256": "synthetic", "msu_reader_sha256": "synthetic"}
            expected = {**config, "model": {"variant": "pretrained_spatial"},
                        "execution": {"source_training": False, "source_decode": True, "source_scoring": True},
                        "budget": {"max_seconds": BUDGET["max_seconds"],
                                   "max_output_bytes": BUDGET["max_output_bytes"], "max_optimizer_steps": None}}
            plan = {"scope": "MSU_frozen_matched_crop_development_diagnostic", "authority": AUTHORITY,
                    "budget": BUDGET, "policy": REAL_PIXEL_POLICY, "config": expected,
                    "implementation_sha256": {}, "preview_root": str(root), "cohort": {},
                    "approval_sha256": digest(root / "visual-approval.json"),
                    "entries": [{"video": row.to_dict(), "samples": [], "receipt_sha256": "synthetic"} for row in rows],
                    "source_manifest_sha256": MANIFEST_SHA,
                    "bootstrap": {"repetitions": 2000, "seed": addressed_seed(SEED, "MSU-MFSD", "crop-ablation-bootstrap")},
                    **{k: preview[k] for k in ("fas_root", "source_manifest", "preprocessing_sha256", "msu_reader_sha256")}}
            path = root / "plan.json"
            write_json(path, plan)
            with patch("diffusionfas.crop_ablation.approved_preview", return_value=(preview, rows)), \
                    patch("diffusionfas.crop_ablation.full_middle_samples", return_value=[]), \
                    patch("diffusionfas.crop_ablation.source_media_entry", return_value={"receipt_sha256": "synthetic"}), \
                    patch("diffusionfas.crop_ablation.read_json", side_effect=lambda path, *args: read_json(path)):
                self.assertEqual(len(checked_inputs(path, digest(path))["entries"]), 22)
                plan["entries"][0]["video"]["timestamps"] = (*rows[0].timestamps[:-1], rows[0].timestamps[-1] + .01)
                changed = root / "changed.json"
                write_json(changed, plan)
                with self.assertRaisesRegex(ValueError, "retain every eligible"):
                    checked_inputs(changed, digest(changed))

    def fixture(self, root):
        media, frames, row = preview_tests.CropPreviewTests().make_media(root)
        cache = root / "cache.npz"
        points = POINTS * .3 + [45, 25]
        np.savez(cache, frames=np.stack([cv2.resize(f, (256, 256), interpolation=cv2.INTER_CUBIC)
                                        for f in frames]),
                 timestamps=row.timestamps, native_timestamps=row.timestamps,
                 native_orders=np.arange(17), boxes=np.tile([0., 0., 128., 96.], (17, 1)),
                 landmarks=np.tile(points, (17, 1, 1)))
        return media, frames, replace(row, dataset="MSU-MFSD", cache_path=str(cache), cache_sha256=digest(cache))

    def test_all_sixteen_samples_use_existing_lower_middle_not_alternate_window(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, row = self.fixture(Path(directory))
            samples = full_middle_samples(row)
            self.assertEqual([s["clip_slot"] for s in samples], list(range(16)))
            self.assertEqual([s["native_order"] for s in samples], list(range(16)))
            self.assertEqual([s["cache_index"] for s in samples], list(range(16)))
            self.assertEqual([s["native_timestamp"] for s in samples], list(row.timestamps[:16]))

    def test_pair_renderer_and_tensor_normalization_match_locked_legacy_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            _, frames, row = self.fixture(Path(directory))
            module = SimpleNamespace(shared_context_crop=lambda rgb, box, points: (rgb, {}))
            pair, evidence = render_pair(row, full_middle_samples(row), dict(enumerate(frames)), module)
            expected = ClipCache(256).get(row)
            actual = torch.from_numpy(pair["legacy"]).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(evidence["legacy_cache_pixel_matches"], 16)
            self.assertEqual(evidence["bilinear_support_verified_frames"], 16)
            self.assertEqual(evidence["padding_fraction"], 0)
            self.assertEqual(pair["real_pixels"].shape, (16, 256, 256, 3))
            with self.assertRaisesRegex(ValueError, "exact fixed T16"):
                render_pair(row, full_middle_samples(row)[1:], dict(enumerate(frames)), module)
            bad = SimpleNamespace(shared_context_crop=lambda rgb, box, points: (np.zeros_like(rgb), {}))
            with self.assertRaisesRegex(ValueError, "legacy crop differs"):
                render_pair(row, full_middle_samples(row), dict(enumerate(frames)), bad)

    def test_actual_residual_draws_are_matched_between_crop_arms(self):
        torch.set_num_threads(1)
        frames = np.zeros((16, 256, 256, 3), dtype=np.uint8)
        pair = {"legacy": frames, "real_pixels": np.full_like(frames, 255)}
        model = ZeroDenoiser().eval()
        profiles = score_pair(model, pair, "same-original", {"seed": SEED, "scoring": SCORING},
                              torch.device("cpu"))
        np.testing.assert_array_equal(profiles["legacy"], profiles["real_pixels"])
        replay = score_pair(model, pair, "same-original", {"seed": SEED, "scoring": SCORING},
                            torch.device("cpu"))
        self.assertEqual(profiles, replay)
        other = score_pair(model, pair, "other-original", {"seed": SEED, "scoring": SCORING},
                           torch.device("cpu"))
        self.assertNotEqual(profiles, other)
        with self.assertRaisesRegex(ValueError, "both fixed"):
            score_pair(model, {"legacy": frames}, "key", {}, torch.device("cpu"))

    def test_paired_auroc_delta_has_fixed_direction_and_same_population(self):
        legacy = [Prediction(f"v{i}", "MSU-MFSD", f"s{i % 3}", "validation",
                             "bona_fide" if i < 6 else "attack", float(i)) for i in range(22)]
        candidate = [replace(r, score=-r.score) for r in legacy]
        report = compare_predictions(dict(zip(ARMS, (legacy, candidate))), {"repetitions": 100, "seed": 42})
        self.assertEqual(report["arms"]["legacy"]["auroc_usable"], 1)
        self.assertEqual(report["arms"]["real_pixels"]["auroc_usable"], 0)
        self.assertEqual(report["real_pixels_minus_legacy"], -1)
        self.assertEqual(report["paired_subject_bootstrap"]["declared_groups"], 3)
        with self.assertRaisesRegex(ValueError, "identical ordered"):
            compare_predictions(dict(zip(ARMS, (legacy, list(reversed(candidate))))),
                                {"repetitions": 100, "seed": 42})

    def test_worker_preserves_all_originals_and_shared_failures_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media, frames, template = self.fixture(root)
            fas = root / "fas"
            source = fas / "src/fas/preprocessing.py"
            source.parent.mkdir(parents=True)
            source.write_text("# synthetic fixture\n")
            reader = fas / "scripts/inventory_msu.py"
            reader.parent.mkdir(parents=True)
            reader.write_text("# synthetic fixture\n")
            output = root / "ablation"
            output.mkdir()
            entries = []
            for i in range(22):
                row = replace(template, key=f"MSU-MFSD/v{i}", group=f"s{i % 3}",
                              label="bona_fide" if i < 6 else "attack")
                entries.append({"video": row.to_dict(), "member": "fixture.avi",
                                "receipt_sha256": "synthetic", "samples": full_middle_samples(row)})
            plan = {"entries": entries, "scope": "synthetic", "fas_root": str(fas),
                    "preprocessing_sha256": digest(source), "msu_reader_sha256": digest(reader),
                    "source_manifest": "unused", "config": {},
                    "bootstrap": {"repetitions": 100, "seed": 42}}
            path = output / "plan.json"
            write_json(path, plan)
            module = SimpleNamespace(__file__=str(source),
                                     shared_context_crop=lambda rgb, box, points: (rgb, {}))

            def stage(entry, fas, data, target, stack, context, deadline, **kwargs):
                self.assertTrue(kwargs["allow_validation_attacks"])
                if entry is entries[0]:
                    raise MediaFailure("synthetic_decode_failure")
                shutil.copyfile(media, target)

            def replay(path, consume, timeout):
                for index, frame in enumerate(frames):
                    consume(index, frame)

            def scores(model, pair, key, config, device):
                return {"legacy": [.1, .2, .3], "real_pixels": [.2, .3, .4]}

            with patch("diffusionfas.crop_ablation.checked_inputs", return_value=plan), \
                    patch("diffusionfas.crop_ablation.importlib",
                          SimpleNamespace(import_module=lambda name: module)), \
                    patch("diffusionfas.crop_ablation.decoder_policy",
                          return_value=({"MSU-MFSD": "fas_strict_ffmpeg"}, "synthetic")), \
                    patch("diffusionfas.crop_ablation.FasNativeReplay") as native, \
                    patch("diffusionfas.crop_ablation.stage_member", side_effect=stage) as stager, \
                    patch("diffusionfas.crop_ablation.configure_gpu", return_value={"synthetic": True}), \
                    patch("diffusionfas.crop_ablation.build_model", return_value=ZeroDenoiser()), \
                    patch("diffusionfas.crop_ablation.score_pair", side_effect=scores) as scorer, \
                    patch("diffusionfas.crop_ablation.torch.cuda.reset_peak_memory_stats"), \
                    patch("diffusionfas.crop_ablation.torch.cuda.max_memory_allocated", return_value=0), \
                    patch("diffusionfas.crop_ablation.torch.cuda.max_memory_reserved", return_value=0):
                native.return_value.bind.return_value = replay
                run_ablation(path, digest(path))
                self.assertEqual(stager.call_count, 22)
                self.assertEqual(scorer.call_count, 21)
                with self.assertRaisesRegex(ValueError, "already attempted"):
                    run_ablation(path, digest(path))
            result = read_json(output / "result.json")
            for arm in ARMS:
                counts = result["comparison"]["arms"][arm]["coverage"]
                self.assertEqual(counts["bona_fide"], {"originals": 6, "usable": 5, "technical_failures": 1})
                self.assertEqual(counts["attack"]["usable"], 16)
            first = read_json(output / "records/score-00.json")
            self.assertTrue(all(r["score"] is None and r["failure"] == "synthetic_decode_failure"
                                for r in first["arms"].values()))
            self.assertEqual(list((output / "staging").iterdir()), [])
            report = summarize(path, {"completed": True})
            self.assertTrue(report["completed"])
            self.assertNotIn("records_sha256", report["result"])
            self.assertEqual(result["training_runs"], 0)
