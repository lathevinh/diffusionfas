"""Metadata-only preparation and explicit reuse of completed spatial fits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from .io import digest, read_json, require_private_output, write_json
from .spatial_dino import ARMS, SEED, SETTINGS, validate_state
from .spatial_preparation import code_pins

ORIGIN_SHA = "866233a31917bae7d73e943a533a72916edb15383ad13bd1af675259fc2c896f"
ORIGIN_APPROVAL_SHA = "9dc6c802efcb98fb9313fe4fd8c70d0978d53a08d7ed495e282a4ff58eea201b"
REUSED = tuple(("OULU-NPU", arm) for arm in ARMS) + (("CASIA-FASD", "reference"),)
ANSWER = "approve_exact_spatial_dino_recovery_6_reused_14_new_20_calibrators_14GiB_120h_no_retry"
CHANGED_CODE = {
    "src/diffusionfas/spatial_study.py", "src/diffusionfas/spatial_verify.py",
    "scripts/spatial_dino.py", "tests/test_spatial_study.py",
}
ADDED_CODE = {"src/diffusionfas/spatial_recovery.py", "tests/test_spatial_recovery.py"}


def contract() -> tuple[dict, dict, dict]:
    from .spatial_study import BUDGET, EXECUTION_POLICY, SCOPE
    return (
        {**BUDGET, "max_gpu_bytes": 14*1024**3},
        {**SCOPE, "new_backbone_fits": 14, "reused_backbone_fits": 6,
         "new_calibrator_fits": 20},
        {**EXECUTION_POLICY,
         "recovery": "six_completed_origin_fits_only_partial_fit_restarted_from_generic_weights",
         "failure": "new_separately_approved_attempt_no_automatic_retry_or_resume",
         "inherited_results_selection": "fixed_completed_inventory_not_target_performance",
         "inherited_checkpoint_checks": "CPU_history_hash_now_GPU_source_validation_and_full_replay_after_approval"},
    )


def reuse_entry(plan: dict, target: str, arm: str) -> dict | None:
    if "recovery" not in plan:
        return None
    return next((entry for entry in plan["recovery"]["reused"]
                 if (entry["target"], entry["arm"]) == (target, arm)), None)


def validate_checkpoint(checkpoint: Path, trace: Path, fitting_count: int, arm: str,
                        guard: Callable) -> dict:
    guard()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_state(state, fitting_count, arm)
    if state["trace_sha256"] != digest(trace):
        raise ValueError("inherited spatial trace differs from checkpoint")
    if not state["state_dict"] or any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state["state_dict"].values()
    ):
        raise ValueError("finite inherited spatial tensors required")
    from .spatial_study import verify_paired_traces
    verify_paired_traces([trace], [state["optimizer_steps"]], guard)
    return state


def origin_inventory(path: Path, guard: Callable) -> tuple[dict, dict, dict]:
    from .spatial_study import ANSWER as ORIGINAL_ANSWER, BUDGET, EXECUTION_POLICY, SCOPE
    from .spatial_study import REPOSITORY, execution_runtime, verify_inputs, verify_paired_traces
    old = read_json(path, ORIGIN_SHA)
    root = path.parent
    approval = read_json(root/"approval.json", ORIGIN_APPROVAL_SHA)
    supervision = read_json(root/"supervision.json")
    terminal = read_json(root/"terminal.json")
    if (old["study"] != "spatial_dino_execution_v1" or old["budget"] != BUDGET
            or old["execution_scope"] != SCOPE or old["execution_policy"] != EXECUTION_POLICY
            or old["settings"] != SETTINGS or old["runtime"] != execution_runtime()
            or approval["answer"] != ORIGINAL_ANSWER or approval["plan_sha256"] != ORIGIN_SHA
            or approval["budget"] != BUDGET or approval["scope"] != SCOPE
            or approval["execution_policy"] != EXECUTION_POLICY
            or terminal != {"completed": False, "plan_sha256": ORIGIN_SHA,
                            "scientific_success_claimed": False,
                            "supervision_sha256": digest(root/"supervision.json")}
            or supervision["completed"] is not False
            or supervision["guard_failure"] != "conservative_device_gpu_memory_cap"):
        raise PermissionError("exact approved GPU-guard-failed original attempt required")
    if any((root/name).exists() for name in ("completion.json", "aggregate.json", "verification.json")):
        raise ValueError("failed origin must not claim scientific completion")
    current = code_pins(REPOSITORY)
    historical = old["code_sha256"]
    if (set(current)-set(historical) != ADDED_CODE or set(historical)-set(current)
            or any(current[name] != sha for name, sha in historical.items() if name not in CHANGED_CODE)):
        raise PermissionError("recovery may change controller code only, not scientific recipe/model code")
    verify_inputs(old, guard)
    bindings = {**old["files"], str(path.resolve()): ORIGIN_SHA}
    for name in ("approval.json", "attempt.json", "started.json", "supervision.json", "terminal.json", "batch.log"):
        bindings[str((root/name).resolve())] = digest(root/name)
    expected_checkpoints = {root/"models"/f"{target}-{arm}"/"model.pth" for target, arm in REUSED}
    if set((root/"models").glob("*/model.pth")) != expected_checkpoints:
        raise ValueError("exact six completed origin checkpoints required")
    entries = []
    for target, arm in REUSED:
        folder = root/"models"/f"{target}-{arm}"
        start = read_json(folder/"started.json")
        if start != {"target": target, "arm": arm, "seed": SEED, "plan_sha256": ORIGIN_SHA}:
            raise ValueError("original training start binding differs")
        if read_json(folder/"calibrator-started.json") != {"plan_sha256": ORIGIN_SHA}:
            raise ValueError("completed original fit/calibration stage required")
        inventory = next(row for row in old["inventory"] if row["target"] == target and row["arm"] == arm)
        checkpoint, trace = folder/"model.pth", folder/"batches.jsonl"
        state = validate_checkpoint(checkpoint, trace, inventory["fitting_usable"], arm, guard)
        entry = {"target": target, "arm": arm, "checkpoint": str(checkpoint.resolve()),
                 "checkpoint_sha256": digest(checkpoint), "trace": str(trace.resolve()),
                 "trace_sha256": digest(trace), "selected_epoch": state["selected_epoch"],
                 "optimizer_steps": state["optimizer_steps"],
                 "history_epochs": len(state["history"])}
        entries.append(entry)
        del state
        for name in ("started.json", "calibrator-started.json", "model.pth", "batches.jsonl"):
            bindings[str((folder/name).resolve())] = digest(folder/name)
    verify_paired_traces([Path(e["trace"]) for e in entries[:5]],
                        [e["optimizer_steps"] for e in entries[:5]], guard)
    partial = root/"models"/"CASIA-FASD-uniform"
    if (not (partial/"started.json").is_file() or not (partial/"batches.jsonl").is_file()
            or (partial/"model.pth").exists() or (partial/"calibrator-started.json").exists()):
        raise ValueError("expected interrupted CASIA uniform fit required; never import partial training")
    for name in ("started.json", "batches.jsonl"):
        bindings[str((partial/name).resolve())] = digest(partial/name)
    recovery = {
        "origin_plan": str(path.resolve()), "origin_plan_sha256": ORIGIN_SHA,
        "origin_approval_sha256": ORIGIN_APPROVAL_SHA,
        "origin_supervision_sha256": digest(root/"supervision.json"),
        "origin_failed_wall_seconds": supervision["wall_seconds"],
        "reused": entries, "discarded_partial_fit": {"target": "CASIA-FASD", "arm": "uniform"},
        "new_fits": 14, "new_calibrators": 20, "final_checkpoint_replays": 20,
        "old_attempt_remains_failed": True, "old_approval_does_not_authorize_recovery": True,
        "PAD_crop_decodes_during_preparation": 0, "GPU_checkpoint_replay_during_preparation": False,
    }
    return old, recovery, bindings


def verify_recovery_inputs(plan: dict, guard: Callable) -> None:
    old, recovery, bindings = origin_inventory(Path(plan["recovery"]["origin_plan"]), guard)
    if plan["recovery"] != recovery or plan["files"] != bindings:
        raise ValueError("exact inherited checkpoint/provenance bindings required")
    for field in ("settings", "policy", "entries", "contexts", "inventory", "counts", "groups",
                  "runtime", "fas_root", "preparation_path", "preparation_sha256", "budget_estimate"):
        if plan[field] != old[field]:
            raise ValueError(f"recovery changed original scientific inputs: {field}")
    for filename, sha in bindings.items():
        guard()
        if digest(Path(filename)) != sha:
            raise ValueError(f"recovery bound file changed: {Path(filename).name}")


def seal(origin_path: Path, output: Path) -> Path:
    from .spatial_study import REPOSITORY, preparation_summary
    require_private_output(output/"plan.json")
    if output.exists():
        raise FileExistsError("new immutable spatial recovery directory required")
    begun = time.monotonic()

    def guard() -> None:
        if time.monotonic()-begun > 1800:
            raise TimeoutError("recovery metadata preparation ceiling")

    old, recovery, bindings = origin_inventory(origin_path, guard)
    budget, scope, policy = contract()
    plan = {**old, "study": "spatial_dino_recovery_v1", "budget": budget,
            "execution_scope": scope, "execution_policy": policy,
            "code_sha256": code_pins(REPOSITORY), "files": bindings, "recovery": recovery}
    path = output/"plan.json"
    write_json(path, plan)
    write_json(output/"preparation.json", {
        **preparation_summary(plan, digest(path)), "recovery": recovery,
        "reused_CPU_histories_and_finite_tensors_checked": 6,
        "neural_parity_and_selected_loss_pending_authorized_replay": True,
    })
    return path


def reuse_receipt(entry: dict, sha: str) -> dict:
    return {"plan_sha256": sha, "origin_plan_sha256": ORIGIN_SHA,
            "target": entry["target"], "arm": entry["arm"], "seed": SEED,
            "checkpoint_sha256": entry["checkpoint_sha256"], "trace_sha256": entry["trace_sha256"]}


def materialize(entry: dict, folder: Path, sha: str, fitting_count: int, guard: Callable) -> dict:
    for name, source, expected in (("model.pth", entry["checkpoint"], entry["checkpoint_sha256"]),
                                   ("batches.jsonl", entry["trace"], entry["trace_sha256"])):
        guard()
        source_path = Path(source)
        if digest(source_path) != expected:
            raise ValueError("inherited checkpoint/trace drift before copying")
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder/name
        with source_path.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024*1024)
        if digest(destination) != expected:
            raise ValueError("inherited checkpoint/trace copy mismatch")
    state = validate_checkpoint(folder/"model.pth", folder/"batches.jsonl", fitting_count, entry["arm"], guard)
    write_json(folder/"reused.json", reuse_receipt(entry, sha))
    return state
