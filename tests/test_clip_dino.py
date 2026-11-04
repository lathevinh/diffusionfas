from __future__ import annotations

import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import time
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusionfas.clip_dino import (
    ARMS, DIMENSION, FRAMES, SEED, SETTINGS, balanced_weights, fit_head, initial_head,
    logits, middle_window, shuffle_indices, shuffled_features,
)
from diffusionfas.clip_study import (
    ANSWER, BUDGET, POLICY, ROLES, SCOPE, clip_runtime, completed_summary, prepare_clip, require_approval, run_worker,
    temporal_groups,
)
from diffusionfas.clip_verify import independent_points, independent_threshold, verify_results
from diffusionfas.gpu_probe import tensor_digest
from diffusionfas.io import digest, read_json, write_json
from diffusionfas.preprocessing import MediaFailure
from diffusionfas.regional_study import source_services
from diffusionfas.source_probe import stage_member
from diffusionfas.supervision import run_bounded

FAS = Path("/mnt/e/FAS")


class ClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_sampling_is_unique_middle_two_seconds(self):
        times = tuple(i/24 for i in range(120))
        chosen = middle_window(times)
        self.assertEqual(len(chosen), 16)
        self.assertEqual(len(set(chosen)), 16)
        self.assertTrue(all(b > a for a, b in zip(chosen, chosen[1:])))
        self.assertAlmostEqual((times[chosen[0]]+times[chosen[-1]])/2, 2.5, delta=.08)
        with self.assertRaisesRegex(ValueError, "insufficient_duration"):
            middle_window((0., .2))
        with self.assertRaises(ValueError):
            middle_window(tuple(i*.2 for i in range(20)))

    def test_shuffle_is_original_addressed_and_order_independent_of_batch(self):
        keys = ["a/one", "a/two", "b/one"]
        first = shuffle_indices(keys, "target")
        reverse = shuffle_indices(keys[::-1], "target")[::-1]
        np.testing.assert_array_equal(first, reverse)
        for row in first:
            np.testing.assert_array_equal(np.sort(row), np.arange(FRAMES))
            self.assertFalse(np.array_equal(row, np.arange(FRAMES)))
        self.assertFalse(np.array_equal(first, shuffle_indices(keys, "another-target")))
        x = np.arange(3*FRAMES*DIMENSION, dtype=np.float32).reshape(3, FRAMES, DIMENSION)
        moved = shuffled_features(x, keys, "target")
        np.testing.assert_array_equal(x.mean(1), moved.mean(1))
        for i in range(3):
            np.testing.assert_array_equal(moved[i], x[i, first[i]])
        with self.assertRaises(ValueError):
            shuffle_indices(["same", "same"], "target")

    def test_architecture_controls_and_order_sensitivity(self):
        a, b = initial_head("temporal"), initial_head("shuffled")
        self.assertEqual(tensor_digest(a.state_dict()), tensor_digest(b.state_dict()))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 31745)
        self.assertEqual(tensor_digest(initial_head("single").state_dict()),
                         tensor_digest(initial_head("mean").state_dict()))
        x = torch.from_numpy(np.random.default_rng(42).normal(size=(3, FRAMES, DIMENSION)))
        with torch.no_grad():
            mean = initial_head("mean")
            torch.testing.assert_close(mean(x), mean(x.flip(1)), rtol=0, atol=1e-14)
            self.assertFalse(torch.allclose(a(x), a(x.flip(1)), rtol=0, atol=1e-12))
            repeated = x[:, :1].repeat(1, FRAMES, 1)
            torch.testing.assert_close(a(repeated), a(repeated.flip(1)), rtol=0, atol=0)

    def test_balanced_loss_and_real_head_fit_selection(self):
        y = np.array([0, 0, 1, 0, 1])
        w = balanced_weights(y, ["a", "a", "a", "b", "b"])
        np.testing.assert_allclose(w, [.125, .125, .25, .25, .25])
        with self.assertRaises(ValueError):
            balanced_weights(np.array([0, 1]), ["a", "b"])
        rng = np.random.default_rng(99)
        x = rng.normal(size=(8, FRAMES, DIMENSION)).astype(np.float32)
        labels = np.tile([0, 1], 4)
        domains = ["a", "a", "b", "b"]*2
        for arm in ARMS:
            model, state = fit_head(arm, x, labels, domains, x, labels, domains, lambda: None, epochs=3)
            values = [r["validation_BCE"] for r in state["history"]]
            self.assertEqual(state["selected_epoch"], int(np.argmin(values))+1)
            self.assertEqual(state["validation_BCE"], min(values))
            np.testing.assert_equal(logits(model, x, lambda: None).shape, (8,))
            replay = initial_head(arm)
            replay.load_state_dict(state["state_dict"], strict=True)
            np.testing.assert_array_equal(logits(model, x, lambda: None), logits(replay, x, lambda: None))

    def test_temporal_group_closure_and_cross_role_content_hard_stop(self):
        rows = [{"dataset": "a", "group_id": str(i), "subject_id": None, "selected_frame": None}
                for i in range(3)]
        entries = [{"key": f"a/{i}", "frames": [{"rgb_sha256": value}]}
                   for i, value in enumerate(("shared", "shared", "other"))]
        roles = {e["key"]: "fitting" for e in entries}
        groups = temporal_groups(rows, entries, roles)
        self.assertEqual(groups["a/0"], groups["a/1"])
        self.assertNotEqual(groups["a/0"], groups["a/2"])
        roles["a/1"] = "evaluation"
        with self.assertRaisesRegex(ValueError, "crosses role"):
            temporal_groups(rows, entries, roles)

    def test_approval_scope_tampering_and_clock_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"plan.json"
            write_json(path, {"fixture": True})
            sha = digest(path)
            approval = {"answer": ANSWER, "plan_sha256": sha, "scope": SCOPE,
                        "policy": POLICY, "budget": BUDGET, "automatic_retry": False,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        "approved_monotonic": time.monotonic()}
            write_json(root/"approval.json", approval)
            with patch("diffusionfas.clip_study.checked_plan", return_value={"fixture": True}):
                require_approval(path, sha, root/"approval.json")
                bad = copy.deepcopy(approval)
                bad["scope"]["encoder_training"] = True
                write_json(root/"bad.json", bad)
                with self.assertRaises(PermissionError):
                    require_approval(path, sha, root/"bad.json")
                expired = {**approval, "approved_monotonic": time.monotonic()-BUDGET["max_seconds"]}
                write_json(root/"expired.json", expired)
                with self.assertRaises(TimeoutError):
                    require_approval(path, sha, root/"expired.json")

    def test_source_staging_scope_was_not_widened_by_helper_extraction(self):
        item = {"video": {"dataset": "MSU-MFSD", "label": "attack", "partition": "evaluation"},
                "member": "safe.mp4"}
        with self.assertRaisesRegex(ValueError, "restricted"):
            stage_member(item, FAS, FAS, Path("/unused"), None, {}, time.monotonic()+1)

    def test_existing_onnxruntime_module_version_does_not_require_a_distribution_alias(self):
        import onnxruntime
        self.assertEqual(clip_runtime()["onnxruntime"], onnxruntime.__version__)

    def test_supervisor_root_rss_includes_controller_without_changing_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("diffusionfas.supervision.process_tree_rss", return_value=2048) as sample:
                report = run_bounded([sys.executable, "-c", "import time; time.sleep(.25)"],
                                     cwd=root, owned_root=root, log=root/"log",
                                     max_seconds=5, max_rss_bytes=4096, max_disk_bytes=1024**2,
                                     rss_root_pid=os.getpid())
            self.assertTrue(report["completed"])
            self.assertTrue(sample.call_count)
            self.assertTrue(all(call.args == (os.getpid(),) for call in sample.call_args_list))
            self.assertEqual(report["sampled_tree_peak_rss_bytes"], 2048)

    @unittest.skipUnless((FAS/"src/fas/preprocessing.py").is_file(), "optional FAS integration")
    def test_actual_crop_contract_and_missing_track_is_a_shared_failure(self):
        services = source_services(FAS)
        import fas.preprocessing as preprocessing

        class Provider:
            module = preprocessing

            def detect(self, rgb):
                return (np.array([20., 15., 80., 90.]),
                        np.array([[35., 35.], [65., 35.], [50., 50.], [40., 70.], [60., 70.]]))

        frames = [np.random.default_rng(i).integers(0, 256, (100, 100, 3), dtype=np.uint8) for i in range(FRAMES)]
        provider = Provider()
        crops, tracking = prepare_clip(frames, provider, lambda: None)
        self.assertEqual(crops.shape, (FRAMES, 224, 224, 3))
        import cv2
        for i, rgb in enumerate(frames):
            box, points = provider.detect(rgb)
            expected, geometry = preprocessing.shared_context_crop(rgb, box, points)
            np.testing.assert_array_equal(crops[i], cv2.resize(expected, (224, 224), interpolation=cv2.INTER_CUBIC))
            self.assertEqual(tracking["geometry"][i], geometry)
        with patch.object(provider, "detect", return_value=None):
            with self.assertRaisesRegex(MediaFailure, "missing_track_endpoint"):
                prepare_clip(frames, provider, lambda: None)
        self.assertIn("study", services)

    def test_strict_threshold_ties_and_original_failure_denominators(self):
        rows = [{"dataset": "a", "binary_label": label, "detector_status": status}
                for label, status in (("attack", "success"), ("attack", "success"),
                                      ("attack", "failed"), ("bona_fide", "success"), ("bona_fide", "failed"))]
        scores = np.array([.5, .5, .1])
        self.assertEqual(independent_threshold(scores, [r for r in rows if r["detector_status"] == "success"],
                                               Fraction(1, 20)), .5)
        point = independent_points(scores, rows, .5, Fraction(1, 20))
        self.assertEqual(point["accepted_attack"], 0)
        self.assertEqual(point["BFA"], .5)
        self.assertEqual(point["AUROC_usable"], 1.)
        self.assertTrue(point["utility_feasible"])

    @unittest.skipUnless((FAS/"scripts/paper321/study.py").is_file(), "optional FAS integration")
    def test_complete_synthetic_producer_and_saved_feature_reconstruction(self):
        services = source_services(FAS)
        domains = ("OULU-NPU", "CASIA-FASD", "MSU-MFSD", "SiW-Mv2")
        rows, roles = [], {}
        for domain in domains:
            for role in ROLES:
                for i in range(16 if role == "evaluation" else 4):
                    key = f"{domain}/{role}-{i}"
                    rows.append({"dataset": domain, "video_id": key.split("/", 1)[1],
                                 "binary_label": "attack" if i % 2 else "bona_fide",
                                 "detector_status": "success", "failure_kind": None,
                                 "subject_id": None if domain == "SiW-Mv2" else key,
                                 "group_id": key, "selected_frame": {"rgb_sha256": key},
                                 "final_k1_action": "score"})
                    roles[key] = role
        entries = [{"key": f"{r['dataset']}/{r['video_id']}", "failure": None,
                    "native_frame_count": FRAMES} for r in rows]
        entries[-1]["failure"] = "historical_test_failure"
        contexts = [{"target": domain, "seed": SEED, **{
            role: [key for key, original_role in roles.items() if original_role == role
                   and ((key.startswith(domain+"/")) if role == "evaluation" else not key.startswith(domain+"/"))]
            for role in ROLES}} for domain in domains]
        plan = {"entries": entries, "contexts": contexts, "fas_root": str(FAS),
                "groups": {e["key"]: e["key"] for e in entries},
                "checkpoints": {d: {"sha256": "synthetic-only"} for d in domains}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"plan.json", {"synthetic_fixture_only": True})
            sha = digest(root/"plan.json")
            write_json(root/"attempt.json", {"supervisor_pid": os.getppid(), "plan_sha256": sha})
            write_json(root/"approval.json", {"synthetic_fixture_only": True})
            approval = {"approved_monotonic": time.monotonic()}

            def extract_fixture(plan, root, guard):
                rng = np.random.default_rng(72)
                statuses = {e["key"]: {"success": e["failure"] is None, **(
                    {} if e["failure"] is None else {"failure": e["failure"]})} for e in entries}
                statuses[entries[-2]["key"]] = {"success": False, "failure": "synthetic_tracking_failure"}
                receipt = {"statuses": statuses, "positions": {}, "files": {}, "synthetic_encoder_frames": 8,
                           "exact_synthetic_replay": True, "production_encoder_frames": 0,
                           "canonical_native_frames_verified": 0, "decode_sha256": {}, "tracking_sha256": {}}
                for i, entry in enumerate(entries):
                    if entry["failure"]:
                        continue
                    token = f"{i:05d}"
                    decoded = root/"decode"/f"{token}.json"
                    write_json(decoded, {"strict_success": True, "failure": None,
                                         "frame_count": FRAMES, "frames_verified": FRAMES})
                    receipt["decode_sha256"][token] = digest(decoded)
                    receipt["canonical_native_frames_verified"] += FRAMES
                    if statuses[entry["key"]]["success"]:
                        tracking = root/"tracking"/f"{token}.json"
                        write_json(tracking, {"key": entry["key"], "boxes": np.ones((FRAMES, 4)).tolist(),
                                              "landmarks": np.ones((FRAMES, 5, 2)).tolist(), "geometry": [{}]*FRAMES})
                        receipt["tracking_sha256"][token] = digest(tracking)
                for context in contexts:
                    target = context["target"]
                    keys = sorted({key for role in ROLES for key in context[role]})
                    array = np.full((len(keys), FRAMES, DIMENSION), np.nan, np.float32)
                    for i, key in enumerate(keys):
                        if statuses[key]["success"]:
                            array[i] = rng.normal(0, .1, (FRAMES, DIMENSION))
                            label = int(key.rsplit("-", 1)[1]) % 2
                            array[i, :, :64] += 2*label-1
                            receipt["production_encoder_frames"] += FRAMES
                    path = root/f"{target}.npy"
                    with path.open("xb") as stream:
                        np.save(stream, array, allow_pickle=False)
                    receipt["files"][target] = digest(path)
                    receipt["positions"][target] = {key: i for i, key in enumerate(keys)}
                write_json(root/"features.json", receipt)
                return receipt

            verification = {}

            def verify_fixture(*args, **kwargs):
                verification.update(verify_results(plan, root, sha, lambda: None))
                write_json(root/"verification.json", verification)
                write_json(root/"supervision.json", {"completed": True, "plan_sha256": sha})
                summary = completed_summary(plan, root, sha)
                self.assertEqual(len(summary["cases"]), 60)
                text = json.dumps(summary)
                for forbidden in ('"threshold":', '"threshold_interval":',
                                  '"oracle_threshold_not_deployable":', '"coefficients":',
                                  '"keys":', '"probabilities":', '"positions":'):
                    self.assertNotIn(forbidden, text)

            with (patch.dict(SETTINGS, {"epochs": 2}),
                  patch("diffusionfas.clip_study.require_approval", return_value=(plan, approval)),
                  patch("diffusionfas.clip_study.verify_inputs"),
                  patch("diffusionfas.clip_study.inputs", return_value=(services, rows, {"contexts": contexts})),
                  patch("diffusionfas.clip_verify.inputs", return_value=(services, rows, {"contexts": contexts})),
                  patch("diffusionfas.clip_study.extract", side_effect=extract_fixture),
                  patch("diffusionfas.clip_study.subprocess.run", side_effect=verify_fixture)):
                run_worker(root/"plan.json", sha, root/"approval.json")
            self.assertTrue(verification["completed"])
            self.assertEqual(verification["operating_cases"], 60)
            self.assertEqual(verification["conditional_intervals"], 132)
            self.assertEqual(verification["new_fits"], 0)
            self.assertLessEqual(verification["maximum_probability_absolute_error"], 1e-10)
            result = read_json(root/"result.json")
            self.assertEqual(len(result["cases"]), 60)
            target = read_json(root/"context-SiW-Mv2.json")
            self.assertEqual(target["policies"]["temporal"], target["policies"]["temporal_inference_shuffle"])
            self.assertEqual(len(target["keys"]["evaluation"]), 14)


if __name__ == "__main__":
    unittest.main()
