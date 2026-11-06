from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusionfas.io import digest, object_digest, read_json, write_json
from diffusionfas.regional_study import source_services
from diffusionfas.spatial_crop_preparation import encode_square
from diffusionfas.spatial_crop_study import (
    ANSWER, BUDGET, EXECUTION_POLICY, POLICY, PREPARATION_SHA, SETTINGS,
    SCOPE, SYSTEMS, input_binding, loaders, split_system, verify_traces,
    check_probe,
)
from diffusionfas.spatial_dino import (
    SEED, SETTINGS as ENGINE_SETTINGS, SourceModules, source_modules,
)
from diffusionfas.spatial_preparation import PROBE_CAPS, fixture
from diffusionfas.spatial_study import checked_plan, completed_summary, require_approval, worker
from diffusionfas.spatial_verify import verify_results
from test_spatial_dino import TinyReference
import test_spatial_study as fixtures

ROOT = Path("/mnt/e/FAS")


class SpatialCropStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.services = source_services(ROOT)

    def test_fixed_systems_scope_and_no_old_approval(self):
        self.assertEqual(len(SYSTEMS), 6)
        self.assertEqual(SCOPE["backbone_fits"], 24)
        self.assertEqual(SCOPE["operating_cases"], 144)
        self.assertEqual(SCOPE["conditional_intervals"], 360)
        for system in SYSTEMS:
            self.assertEqual("_".join(split_system(system)), system)
        with self.assertRaises(ValueError):
            split_system("square_permuted")
        baseline = {"study": "spatial_crop_execution_v1", "crop_study": {},
                    "execution_scope": SCOPE, "execution_policy": EXECUTION_POLICY,
                    "budget": BUDGET, "settings": SETTINGS, "policy": POLICY,
                    "code_sha256": {}, "runtime": {},
                    "scientific_execution_authorized": False,
                    "executable_scientific_runner_available": True, "execution_runner_sealed": True}
        with tempfile.TemporaryDirectory() as directory, patch(
            "diffusionfas.spatial_study.code_pins", return_value={}
        ), patch("diffusionfas.spatial_study.execution_runtime", return_value={}):
            path = Path(directory)/"plan.json"
            write_json(path, baseline)
            checked_plan(path, digest(path))
            wrong = {**baseline, "recovery": {}}
            write_json(path.parent/"wrong.json", wrong)
            with self.assertRaises(PermissionError):
                checked_plan(path.parent/"wrong.json", digest(path.parent/"wrong.json"))
            approval = {"answer": fixtures.ANSWER, "plan_sha256": digest(path),
                        "budget": BUDGET, "scope": SCOPE, "execution_policy": EXECUTION_POLICY,
                        "approved_monotonic": time.monotonic(),
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
            write_json(path.parent/"approval.json", approval)
            with self.assertRaises(PermissionError):
                require_approval(path, digest(path), path.parent/"approval.json")

    def test_probe_requires_bounded_success_positive_timings_and_replay(self):
        result = {"measured_step_seconds": [1., 2.], "same_process_replay_exact": True,
                  "independent_replay_exact": True}
        probe = {"caps": PROBE_CAPS, "synthetic_optimizer_steps": 15, "results": [result]*5}
        supervisor = {"completed": True, "returncode": 0, "guard_failure": None,
                      "caps": {**PROBE_CAPS, "max_gpu_bytes": BUDGET["max_gpu_bytes"]}}
        check_probe(probe, supervisor)
        with self.assertRaises(ValueError):
            check_probe(probe, {**supervisor, "completed": False})
        with self.assertRaises(ValueError):
            check_probe({**probe, "results": [{**result, "measured_step_seconds": [-1., 2.]}]*5}, supervisor)
        with self.assertRaises(ValueError):
            check_probe({**probe, "results": [{**result, "independent_replay_exact": False}]*5}, supervisor)

    def test_explicit_crop_loader_prior_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = [[76, 80], [148, 80], [112, 115], [86, 151], [138, 151]]
            legacy = np.zeros((224, 224, 3), np.uint8)
            square = np.full_like(legacy, 180)
            assets = {"legacy": encode_square(root/"old.png", legacy),
                      "square": encode_square(root/"new.png", square)}
            row = {"dataset": "d", "video_id": "v"}
            entry = {"key": "d/v", "status": "usable", "crops": assets,
                     "landmarks224": {"legacy": points, "square": (np.asarray(points)+2).tolist()}}
            plan = {"crop_study": {}, "entries": [entry]}
            old_rgb, old_prior = loaders(plan, lambda: None, "legacy_reference")
            new_rgb, new_prior = loaders(plan, lambda: None, "square_reference")
            np.testing.assert_array_equal(old_rgb(row), legacy)
            np.testing.assert_array_equal(new_rgb(row), square)
            self.assertFalse(np.array_equal(old_prior(row), new_prior(row)))
            from diffusionfas.spatial_study import loaders as shared_loaders
            with self.assertRaisesRegex(ValueError, "explicit system"):
                shared_loaders(plan, lambda: None)
            (root/"new.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "payload changed"):
                new_rgb(row)
            entry["status"] = "failed"
            with self.assertRaisesRegex(ValueError, "shared failure"):
                old_rgb(row)

    def test_crop_pairing_allows_only_RGB_and_prior_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/f"{system}.jsonl" for system in SYSTEMS]
            records = []
            for i, path in enumerate(paths):
                record = {"epoch": 1, "step": 0, "selected": [1], "donors": [-1],
                          "batch_sha256": "old" if i < 3 else "new",
                          "spatial_prior_sha256": "old_prior" if i < 3 else "new_prior"}
                records.append(record)
                path.write_text(json.dumps(record)+"\n")
            verify_traces(paths, [1]*6, lambda: None)
            for i in range(3, 6):
                records[i]["selected"] = [2]
                paths[i].write_text(json.dumps(records[i])+"\n")
            with self.assertRaisesRegex(ValueError, "pairing mismatch"):
                verify_traces(paths, [1]*6, lambda: None)

    def test_complete_synthetic24_fit_replay_and_two_denominator_views(self):
        actual = source_modules(ROOT)
        stub = ModuleType("synthetic_crop_reference")
        stub.Reference = lambda initialize=True: TinyReference()
        stub.configure = lambda seed: torch.manual_seed(seed)
        stub.tensor_images, stub.losses = actual.model.tensor_images, actual.model.losses

        def check(model):
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise ValueError("invalid synthetic crop gradient")
        stub.check_gradients = check
        source = SourceModules(stub, actual.augmentation)
        rows, contexts, indexed, groups = fixtures.SpatialStudyTests.synthetic_contexts(self)
        entries = [{"key": key, "status": "usable" if row["detector_status"] == "success" else "failed"}
                   for key, row in indexed.items()]
        plan = {"fas_root": str(ROOT), "crop_study": {}, "contexts": contexts, "groups": groups,
                "entries": entries, "preparation_sha256": PREPARATION_SHA,
                "common_eligibility_sha256": object_digest(entries), "execution_scope": SCOPE,
                "execution_policy": {**EXECUTION_POLICY, "bootstrap_repetitions": 40}, "budget": BUDGET}
        _, images, priors = fixture()

        def synthetic_loaders(plan, guard, *, system):
            crop, _ = split_system(system)
            return (lambda row: np.roll(images[row["index"] % 6], int(crop == "square"), axis=1).copy(),
                    lambda row: np.roll(priors[row["index"] % 6], int(crop == "square"), axis=1).copy())

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            path = root/"plan.json"
            write_json(path, {"synthetic": True})
            sha = digest(path)
            approval = {"answer": ANSWER, "plan_sha256": sha, "budget": BUDGET, "scope": SCOPE,
                        "execution_policy": plan["execution_policy"], "approved_monotonic": time.monotonic(),
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
            approval_path = root/"approval.json"
            write_json(approval_path, approval)
            write_json(root/"attempt.json", {"supervisor_pid": os.getppid(), "plan_sha256": sha})
            for module in ("spatial_study", "spatial_verify"):
                prefix = f"diffusionfas.{module}."
                stack.enter_context(patch(prefix+"inputs", return_value=(self.services, rows, {})))
                stack.enter_context(patch(prefix+"population", return_value=indexed))
                stack.enter_context(patch(prefix+"loaders", side_effect=synthetic_loaders))
                stack.enter_context(patch(prefix+"source_modules", return_value=source))
                stack.enter_context(patch(prefix+"DEVICE", "cpu"))
            stack.enter_context(patch("diffusionfas.spatial_study.require_approval", return_value=(plan, approval)))
            stack.enter_context(patch("diffusionfas.spatial_study.verify_inputs"))
            stack.enter_context(patch("torch.cuda.set_per_process_memory_fraction"))
            stack.enter_context(patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=16*1024**3)))
            stack.enter_context(patch("torch.cuda.empty_cache"))
            stack.enter_context(patch.dict(ENGINE_SETTINGS, {"epochs": 2}))

            def reconstruct(*args, **kwargs):
                self.assertIn("_verify", args[0])
                write_json(root/"verification.json", verify_results(plan, root, sha, lambda: None))
                return SimpleNamespace(returncode=0)

            stack.enter_context(patch("diffusionfas.spatial_study.subprocess.run", side_effect=reconstruct))
            worker(path, sha, approval_path)
            verified = read_json(root/"verification.json")
            self.assertEqual(verified["checkpoint_replays"], 24)
            self.assertEqual(verified["execution_new_calibrator_fits"], 24)
            self.assertEqual(verified["operating_cases"], 144)
            self.assertEqual(verified["conditional_intervals"], 360)
            self.assertLessEqual(verified["maximum_probability_absolute_error"], 1e-10)
            summary = completed_summary(root, sha, verified, {"completed": True}, plan=plan)
            self.assertEqual(len(summary["macro"]), 36)
            cases = summary["cases"]
            self.assertEqual({case["view"] for case in cases}, {"original", "common"})
            originals = [case for case in cases if case["target"] == "a" and case["view"] == "original"]
            common = [case for case in cases if case["target"] == "a" and case["view"] == "common"]
            self.assertTrue(all(case["policy"]["original_bona_fide"] == 3 for case in originals))
            self.assertTrue(all(case["policy"]["original_bona_fide"] == 2 for case in common))
            self.assertEqual(len(list((root/"models").glob("*/started.json"))), 24)
            self.assertEqual(len(list((root/"models").glob("*/calibrator-started.json"))), 24)
            self.assertFalse(list((root/"models").glob("*/reused.json")))
            checkpoint = root/"models/a-legacy_reference/model.pth"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(state["input_binding"], input_binding(plan, "a", "legacy_reference"))
            state["input_binding"]["crop"] = "square"
            torch.save(state, checkpoint)
            context_path = root/"context-a.json"
            saved = read_json(context_path)
            saved["models"]["legacy_reference"]["checkpoint_sha256"] = digest(checkpoint)
            context_path.write_text(json.dumps(saved))
            result = read_json(root/"result.json")
            result["artifacts"]["a"] = digest(context_path)
            (root/"result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "input binding mismatch"):
                verify_results(plan, root, sha, lambda: None)


if __name__ == "__main__":
    unittest.main()
