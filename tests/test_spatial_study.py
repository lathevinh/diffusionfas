from __future__ import annotations

from contextlib import ExitStack
import copy
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
from diffusionfas.spatial_dino import (
    ARMS, SEED, SETTINGS, SourceModules, validate_state,
)
from diffusionfas.spatial_preparation import POLICY, fixture
from diffusionfas.spatial_study import (
    ANSWER, BUDGET, EXECUTION_POLICY, ROLES, SCOPE, analyze, approve, checked_plan,
    clock, completed_summary, require_approval, supervise, verify_paired_traces, worker,
)
from diffusionfas.spatial_verify import verify_completed, verify_results
from test_spatial_dino import TinyReference

ROOT = Path("/mnt/e/FAS")


class SpatialStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.services = source_services(ROOT)

    def approval(self, sha):
        return {"answer": ANSWER, "plan_sha256": sha, "budget": BUDGET, "scope": SCOPE,
                "execution_policy": EXECUTION_POLICY, "approved_monotonic": time.monotonic(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}

    def test_exact_owner_plan_and_nonresettable_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"plan.json"
            write_json(path, {"synthetic": True})
            sha = digest(path)
            approval = self.approval(sha)
            write_json(path.parent/"approval.json", approval)
            with patch("diffusionfas.spatial_study.checked_plan", return_value={"synthetic": True}):
                require_approval(path, sha, path.parent/"approval.json")
                for field, value in (("answer", "general_budget_approval"),
                                     ("plan_sha256", "0"*64),
                                     ("scope", {**SCOPE, "backbone_fits": 21}),
                                     ("budget", {**BUDGET, "max_seconds": BUDGET["max_seconds"]+1})):
                    bad = {**approval, field: value}
                    p = path.parent/f"bad-{field}.json"
                    write_json(p, bad)
                    with self.assertRaises(PermissionError):
                        require_approval(path, sha, p)
        with self.assertRaises(TimeoutError):
            clock({**approval, "approved_monotonic": time.monotonic()-BUDGET["max_seconds"]-1})
        with self.assertRaises(TimeoutError):
            clock({**approval, "boot_id": "changed"})
        with self.assertRaises(TimeoutError):
            clock({**approval, "approved_monotonic": time.monotonic()+60})

    def test_checked_plan_rejects_scope_recipe_runtime_and_code_drift(self):
        baseline = {"study": "spatial_dino_execution_v1", "execution_scope": SCOPE,
                    "execution_policy": EXECUTION_POLICY, "budget": BUDGET, "settings": SETTINGS,
                    "policy": POLICY, "code_sha256": {"fixture": "sha"}, "runtime": {"fixture": True},
                    "scientific_execution_authorized": False, "executable_scientific_runner_available": True,
                    "execution_runner_sealed": True}
        with patch("diffusionfas.spatial_study.code_pins", return_value={"fixture": "sha"}), patch(
            "diffusionfas.spatial_study.execution_runtime", return_value={"fixture": True}
        ):
            for field, value in ((None, None), ("execution_scope", {**SCOPE, "backbone_fits": 21}),
                                 ("settings", {**SETTINGS, "epochs": 20}),
                                 ("runtime", {"changed": True}), ("code_sha256", {})):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"plan.json"
                    write_json(path, baseline if field is None else {**baseline, field: value})
                    if field is None:
                        checked_plan(path, digest(path))
                    else:
                        with self.assertRaises(PermissionError):
                            checked_plan(path, digest(path))

    def test_general_instruction_cannot_create_exact_approval_or_reset_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"plan.json"
            write_json(path, {"synthetic": True})
            sha = digest(path)
            with patch("diffusionfas.spatial_study.checked_plan", return_value={}), patch(
                "diffusionfas.spatial_study.verify_inputs"
            ) as verify:
                with self.assertRaises(PermissionError):
                    approve(path, sha, "raise_budget")
                verify.assert_not_called()
                write_json(path.parent/"attempt.json", {"failed": True})
                with self.assertRaises(PermissionError):
                    approve(path, sha, ANSWER)
                verify.assert_not_called()

    def test_supervisor_cannot_resume_and_failed_run_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"plan.json"
            write_json(path, {"synthetic": True})
            sha = digest(path)
            approval = self.approval(sha)
            approval_path = path.parent/"approval.json"
            write_json(approval_path, approval)
            with patch("diffusionfas.spatial_study.require_approval", return_value=({}, approval)), patch(
                "diffusionfas.adapter_pilot.device_memory_bytes", return_value=0
            ), patch("diffusionfas.spatial_study.run_bounded", return_value={
                "completed": False, "returncode": 1, "guard_failure": "test_only"
            }):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    supervise(path, sha, approval_path)
                self.assertFalse(read_json(path.parent/"terminal.json")["completed"])
                with self.assertRaises(FileExistsError):
                    supervise(path, sha, approval_path)
            self.assertFalse((path.parent/"completion.json").exists())

    def test_worker_and_verifier_require_recorded_parent_before_pixel_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"plan.json"
            write_json(path, {"synthetic": True})
            sha = digest(path)
            approval = self.approval(sha)
            approval_path = root/"approval.json"
            write_json(approval_path, approval)
            write_json(root/"attempt.json", {"supervisor_pid": -1, "plan_sha256": sha})
            with patch("diffusionfas.spatial_study.require_approval", return_value=({}, approval)), patch(
                "diffusionfas.spatial_study.verify_inputs"
            ) as verify:
                with self.assertRaises(PermissionError):
                    worker(path, sha, approval_path)
                verify.assert_not_called()
            write_json(root/"started.json", {"worker_pid": -1, "plan_sha256": sha,
                                             "approval_sha256": digest(approval_path)})
            with patch("diffusionfas.spatial_verify.require_approval", return_value=({}, approval)), patch(
                "diffusionfas.spatial_verify.verify_inputs"
            ) as verify:
                with self.assertRaises(PermissionError):
                    verify_completed(path, sha, approval_path)
                verify.assert_not_called()

    def test_five_arm_traces_compare_common_prefixes_and_exact_step_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/f"{arm}.jsonl" for arm in ARMS]
            for i, path in enumerate(paths):
                path.write_text("".join(json.dumps({"step": j})+"\n" for j in range(2+i)))
            verify_paired_traces(paths, [2, 3, 4, 5, 6], lambda: None)
            with self.assertRaises(ValueError):
                verify_paired_traces(paths, [2]*5, lambda: None)
            paths[-1].write_text('{"step": -1}\n')
            with self.assertRaisesRegex(ValueError, "pairing"):
                verify_paired_traces(paths, [2, 3, 4, 5, 1], lambda: None)

    def test_partial_or_mislabeled_checkpoint_history_is_rejected(self):
        state = {"arm": "anatomical", "seed": SEED, "settings_sha256": object_digest(SETTINGS),
                 "history": [{"epoch": 1, "validation": .5}], "selected_epoch": 1,
                 "validation_loss": .5, "optimizer_steps": 1}
        with self.assertRaisesRegex(ValueError, "shortened"):
            validate_state(state, 6, "anatomical")
        for field, value in (("arm", "content"), ("seed", SEED+1), ("settings_sha256", "changed"),
                             ("selected_epoch", 2), ("optimizer_steps", 2)):
            with self.assertRaises(ValueError):
                validate_state({**state, field: value}, 6, "anatomical")
        values = [1., .5]+[.5]*20
        state.update({"history": [{"epoch": i+1, "validation": v} for i, v in enumerate(values)],
                      "optimizer_steps": len(values), "selected_epoch": 2})
        validate_state(state, 6, "anatomical")
        state["history"].append({"epoch": 23, "validation": .5})
        state["optimizer_steps"] += 1
        with self.assertRaisesRegex(ValueError, "beyond patience"):
            validate_state(state, 6, "anatomical")

    def synthetic_contexts(self):
        rows = []
        for domain in ("a", "b", "c", "d"):
            for role in ROLES:
                for rep in range(2):
                    for label in ("bona_fide", "attack"):
                        key = f"{domain}-{role}-{rep}-{label}"
                        rows.append({"dataset": domain, "video_id": key,
                            "group_id": f"{domain}-{role}-{rep}", "subject_id": f"{domain}-{role}-{rep}",
                            "binary_label": label, "detector_status": "success",
                            "selected_frame": {"rgb_sha256": key}, "role": role, "index": len(rows)})
        rows.append({"dataset": "a", "video_id": "failure", "group_id": "a-evaluation-0",
                     "subject_id": "a-evaluation-0", "binary_label": "bona_fide",
                     "detector_status": "failed", "selected_frame": None, "role": "evaluation", "index": len(rows)})
        identity = self.services["fas.source_fitting"].identity
        contexts = [{"target": domain, "seed": SEED, **{
            role: [identity(row) for row in rows if row["role"] == role and
                   (row["dataset"] == domain if role == "evaluation" else row["dataset"] != domain)]
            for role in ROLES}} for domain in ("a", "b", "c", "d")]
        from fas.fusion_study import components
        groups = {}
        for context in contexts:
            originals = [row for row in rows if identity(row) in context["evaluation"]]
            groups.update(components(originals))
        return rows, contexts, {identity(row): row for row in rows}, groups

    def test_complete_synthetic_twenty_model_pipeline_and_tamper_detection(self):
        from diffusionfas.spatial_dino import source_modules
        actual = source_modules(ROOT)
        stub = ModuleType("synthetic_controller_reference")
        stub.Reference = lambda initialize=True: TinyReference()
        stub.configure = lambda seed: torch.manual_seed(seed)
        stub.tensor_images, stub.losses = actual.model.tensor_images, actual.model.losses

        def check(model):
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise ValueError("invalid synthetic controller gradient")
        stub.check_gradients = check
        source = SourceModules(stub, actual.augmentation)
        rows, contexts, indexed, groups = self.synthetic_contexts()
        plan = {"fas_root": str(ROOT), "contexts": contexts, "groups": groups}
        _, images, priors = fixture()
        loaders = (lambda row: images[row["index"] % 6].copy(),
                   lambda row: priors[row["index"] % 6].copy())
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            path = root/"plan.json"
            write_json(path, {"synthetic": True})
            sha = digest(path)
            approval = self.approval(sha)
            approval_path = root/"approval.json"
            write_json(approval_path, approval)
            write_json(root/"attempt.json", {"supervisor_pid": os.getppid(), "plan_sha256": sha})
            for module in ("spatial_study", "spatial_verify"):
                prefix = f"diffusionfas.{module}."
                stack.enter_context(patch(prefix+"inputs", return_value=(self.services, rows, {})))
                stack.enter_context(patch(prefix+"population", return_value=indexed))
                stack.enter_context(patch(prefix+"loaders", return_value=loaders))
                stack.enter_context(patch(prefix+"source_modules", return_value=source))
                stack.enter_context(patch(prefix+"DEVICE", "cpu"))
            stack.enter_context(patch("diffusionfas.spatial_study.require_approval", return_value=(plan, approval)))
            stack.enter_context(patch("diffusionfas.spatial_study.verify_inputs"))
            stack.enter_context(patch("torch.cuda.set_per_process_memory_fraction"))
            stack.enter_context(patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=16*1024**3)))
            stack.enter_context(patch("torch.cuda.empty_cache"))
            stack.enter_context(patch.dict(SETTINGS, {"epochs": 2}))
            stack.enter_context(patch.dict(EXECUTION_POLICY, {"bootstrap_repetitions": 40}))

            def reconstruct(*args, **kwargs):
                self.assertIn("_verify", args[0])
                write_json(root/"verification.json", verify_results(plan, root, sha, lambda: None))
                return SimpleNamespace(returncode=0)

            stack.enter_context(patch("diffusionfas.spatial_study.subprocess.run", side_effect=reconstruct))
            worker(path, sha, approval_path)
            verified = read_json(root/"verification.json")
            self.assertTrue(verified["completed"])
            self.assertEqual(verified["checkpoint_replays"], 20)
            self.assertEqual(verified["operating_cases"], 60)
            self.assertEqual(verified["conditional_intervals"], 132)
            self.assertLessEqual(verified["maximum_probability_absolute_error"], 1e-10)
            summary = completed_summary(root, sha, verified, {"completed": True})
            self.assertEqual(len(summary["macro"]), 15)
            self.assertEqual(summary["scope"]["backbone_fits"], 20)
            result = read_json(root/"result.json")
            result["cases"][0]["policy"]["accepted_attack"] += 1
            # Put the corrupted first case in the first fold so reconstruction
            # rejects it after the first checkpoint, rather than twenty replays.
            with (root/"result.json").open("w") as stream:
                json.dump(result, stream)
            with self.assertRaisesRegex(ValueError, "metric"):
                verify_results(plan, root, sha, lambda: None)


if __name__ == "__main__":
    unittest.main()
