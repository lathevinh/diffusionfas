from __future__ import annotations

from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from diffusionfas.io import digest, object_digest, read_json, write_json
from diffusionfas.spatial_dino import SEED, SETTINGS
from diffusionfas.spatial_recovery import (
    ADDED_CODE, ANSWER, REUSED, contract, materialize, origin_inventory, reuse_entry,
    reuse_receipt, validate_checkpoint, verify_recovery_inputs,
)
from diffusionfas.spatial_study import (
    ANSWER as ORIGINAL_ANSWER, BUDGET, EXECUTION_POLICY, SCOPE, approve, checked_plan,
    require_approval,
)
from diffusionfas.spatial_preparation import POLICY


class SpatialRecoveryTests(unittest.TestCase):
    def state(self, arm):
        losses = [1., .5]+[.5]*20
        return {"arm": arm, "seed": SEED, "settings_sha256": object_digest(SETTINGS),
                "selected_epoch": 2, "validation_loss": .5, "optimizer_steps": len(losses),
                "history": [{"epoch": i+1, "validation": value} for i, value in enumerate(losses)],
                "state_dict": {"weight": torch.zeros(2, 3)}}

    def checkpoint(self, folder, arm):
        folder.mkdir(parents=True, exist_ok=True)
        trace = folder/"batches.jsonl"
        trace.write_text("".join('{"epoch": %d, "step": 0}\n' % i for i in range(1, 23)))
        state = self.state(arm)
        state["trace_sha256"] = digest(trace)
        torch.save(state, folder/"model.pth")
        return state

    def test_only_complete_finite_hash_matched_checkpoints_can_be_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"origin"
            state = self.checkpoint(source, "anatomical")
            checkpoint, trace = source/"model.pth", source/"batches.jsonl"
            entry = {"target": "OULU-NPU", "arm": "anatomical", "checkpoint": str(checkpoint),
                     "checkpoint_sha256": digest(checkpoint), "trace": str(trace),
                     "trace_sha256": digest(trace)}
            plan = {"recovery": {"reused": [entry]}}
            self.assertEqual(reuse_entry(plan, "OULU-NPU", "anatomical"), entry)
            self.assertIsNone(reuse_entry(plan, "CASIA-FASD", "anatomical"))
            destination = root/"new"
            imported = materialize(entry, destination, "new-sha", 6, lambda: None)
            self.assertEqual(imported["selected_epoch"], 2)
            self.assertEqual(digest(destination/"model.pth"), entry["checkpoint_sha256"])
            self.assertEqual(read_json(destination/"reused.json"), reuse_receipt(entry, "new-sha"))
            self.assertFalse((destination/"started.json").exists())
            with self.assertRaises(FileExistsError):
                materialize(entry, destination, "new-sha", 6, lambda: None)
            state["history"] = state["history"][:2]
            state["optimizer_steps"] = 2
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "drift"):
                materialize(entry, root/"drift", "new-sha", 6, lambda: None)
            with self.assertRaisesRegex(ValueError, "shortened"):
                validate_checkpoint(checkpoint, trace, 6, "anatomical", lambda: None)
            state = self.state("anatomical")
            state["trace_sha256"] = digest(trace)
            state["state_dict"]["weight"][0, 0] = float("nan")
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "finite inherited"):
                validate_checkpoint(checkpoint, trace, 6, "anatomical", lambda: None)

    def test_recovery_requires_its_own_exact_budget_scope_and_answer(self):
        budget, scope, policy = contract()
        baseline = {"study": "spatial_dino_recovery_v1", "recovery": {},
                    "execution_scope": scope, "execution_policy": policy, "budget": budget,
                    "settings": SETTINGS, "policy": POLICY, "code_sha256": {"fixture": "sha"},
                    "runtime": {"fixture": True}, "scientific_execution_authorized": False,
                    "executable_scientific_runner_available": True, "execution_runner_sealed": True}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            path = root/"plan.json"
            write_json(path, baseline)
            sha = digest(path)
            stack.enter_context(patch("diffusionfas.spatial_study.code_pins", return_value={"fixture": "sha"}))
            stack.enter_context(patch("diffusionfas.spatial_study.execution_runtime", return_value={"fixture": True}))
            stack.enter_context(patch("diffusionfas.spatial_study.verify_inputs"))
            checked_plan(path, sha)
            with self.assertRaises(PermissionError):
                approve(path, sha, ORIGINAL_ANSWER)
            approval_path = approve(path, sha, ANSWER)
            _, approved = require_approval(path, sha, approval_path)
            self.assertEqual(approved["budget"]["max_gpu_bytes"], 14*1024**3)
            self.assertEqual(approved["scope"]["new_backbone_fits"], 14)
            bad = {**approved, "budget": BUDGET}
            write_json(root/"bad-approval.json", bad)
            with self.assertRaises(PermissionError):
                require_approval(path, sha, root/"bad-approval.json")
            self.assertEqual(scope["backbone_fits"], 20)
            self.assertEqual(scope["new_calibrator_fits"], 20)

    def test_origin_inventory_validates_six_completed_fits_and_excludes_partial(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            historical = {"src/diffusionfas/spatial_study.py": "old", "scientific_model.py": "unchanged"}
            current = {**historical, "src/diffusionfas/spatial_study.py": "new",
                       **{name: "new" for name in ADDED_CODE}}
            old = {"study": "spatial_dino_execution_v1", "budget": BUDGET, "execution_scope": SCOPE,
                   "execution_policy": EXECUTION_POLICY, "settings": SETTINGS, "runtime": {"fixture": True},
                   "code_sha256": historical, "files": {}, "inventory": [
                       {"target": target, "arm": arm, "fitting_usable": 6} for target, arm in REUSED]}
            path = root/"plan.json"
            write_json(path, old)
            sha = digest(path)
            write_json(root/"approval.json", {"answer": ORIGINAL_ANSWER, "plan_sha256": sha,
                "budget": BUDGET, "scope": SCOPE, "execution_policy": EXECUTION_POLICY})
            supervision = {"completed": False, "guard_failure": "conservative_device_gpu_memory_cap",
                           "wall_seconds": 6000}
            write_json(root/"supervision.json", supervision)
            write_json(root/"terminal.json", {"completed": False, "plan_sha256": sha,
                "scientific_success_claimed": False, "supervision_sha256": digest(root/"supervision.json")})
            for name in ("started.json", "attempt.json"):
                write_json(root/name, {"synthetic": True})
            (root/"batch.log").write_text("synthetic original trace\n")
            for target, arm in REUSED:
                folder = root/"models"/f"{target}-{arm}"
                self.checkpoint(folder, arm)
                write_json(folder/"started.json", {"target": target, "arm": arm, "seed": SEED, "plan_sha256": sha})
                write_json(folder/"calibrator-started.json", {"plan_sha256": sha})
            partial = root/"models"/"CASIA-FASD-uniform"
            partial.mkdir()
            write_json(partial/"started.json", {"synthetic": True})
            (partial/"batches.jsonl").write_text('{"epoch": 1, "step": 0}\n')
            stack.enter_context(patch("diffusionfas.spatial_recovery.ORIGIN_SHA", sha))
            stack.enter_context(patch("diffusionfas.spatial_recovery.ORIGIN_APPROVAL_SHA", digest(root/"approval.json")))
            stack.enter_context(patch("diffusionfas.spatial_study.execution_runtime", return_value={"fixture": True}))
            stack.enter_context(patch("diffusionfas.spatial_study.verify_inputs"))
            pins = stack.enter_context(patch("diffusionfas.spatial_recovery.code_pins", return_value=current))
            _, recovery, bindings = origin_inventory(path, lambda: None)
            self.assertEqual(len(recovery["reused"]), 6)
            self.assertNotIn(str(partial/"model.pth"), bindings)
            bad = {**current, "scientific_model.py": "changed"}
            pins.return_value = bad
            with self.assertRaisesRegex(PermissionError, "controller code only"):
                origin_inventory(path, lambda: None)
            pins.return_value = current
            self.checkpoint(partial, "uniform")
            with self.assertRaisesRegex(ValueError, "exact six"):
                origin_inventory(path, lambda: None)

    def test_recovery_cannot_change_population_or_inherited_bytes(self):
        old = {field: {} for field in ("settings", "policy", "entries", "contexts", "inventory", "counts",
               "groups", "runtime", "fas_root", "preparation_path", "preparation_sha256", "budget_estimate")}
        recovery = {"origin_plan": "/synthetic/plan.json", "reused": []}
        plan = {**old, "recovery": recovery, "files": {}}
        with patch("diffusionfas.spatial_recovery.origin_inventory", return_value=(old, recovery, {})):
            verify_recovery_inputs(plan, lambda: None)
            for field in ("contexts", "entries", "groups", "settings"):
                bad = copy.deepcopy(plan)
                bad[field] = {"changed": True}
                with self.assertRaisesRegex(ValueError, "scientific inputs"):
                    verify_recovery_inputs(bad, lambda: None)
            with self.assertRaisesRegex(ValueError, "provenance"):
                verify_recovery_inputs({**plan, "files": {"unapproved": "sha"}}, lambda: None)


if __name__ == "__main__":
    unittest.main()
