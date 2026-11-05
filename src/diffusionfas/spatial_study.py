"""Exactly approved, single-attempt full-backbone spatial DINO execution."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

from .gpu_probe import tensor_digest
from .io import digest, read_json, require_private_output, write_json
from .regional_study import inputs, paired_intervals, runtime, source_state
from .spatial_dino import (
    ARMS, SEED, SETTINGS, SpatialReference, fit, points224, predict, prior_image,
    source_modules, validate_state,
)
from .spatial_preparation import POLICY, PROPOSED_LIMITS, code_pins, importlib_pillow_version
from .supervision import run_bounded, tree_bytes

BUDGET = PROPOSED_LIMITS
ROLES = ("fitting", "validation", "calibration", "evaluation")
CONTRASTS = (
    ("anatomical_minus_content", "anatomical", "content"),
    ("anatomical_minus_permuted", "anatomical", "permuted"),
    ("anatomical_minus_uniform", "anatomical", "uniform"),
    ("anatomical_minus_reference", "anatomical", "reference"),
    ("content_minus_uniform", "content", "uniform"),
    ("uniform_minus_reference", "uniform", "reference"),
)
SCOPE = {
    "targets": 4, "seed": SEED, "backbone_fits": 20, "calibrator_fits": 20,
    "operating_cases": 60, "conditional_intervals": 132,
    "PAD_crop_decode": True, "new_native_video_decode": False, "new_detector": False,
    "generic_pretrained_initialization_only": True, "new_download": False,
    "new_environment": False, "automatic_retry": False, "automatic_expansion": False,
}
EXECUTION_POLICY = {
    "primary_candidate": "anatomical", "reference": "reference",
    "paired_contrasts": [list(item) for item in CONTRASTS],
    "bootstrap_repetitions": 2000, "bootstrap_seed": 20261028,
    "selection": SETTINGS["selection"], "target_model_selection": False,
    "oracle_diagnostic_only": True,
    "shared_clock": "approval_through_input_checks_training_replay_analysis_and_closure",
    "verification": "separate_GPU_checkpoint_replay_no_new_fit_independent_counts_and_bootstrap",
    "failure": "fail_closed_retained_receipts_no_retry_or_resume",
}
ANSWER = "approve_exact_spatial_dino_20_fits_120h_no_retry"
REPOSITORY = Path(__file__).resolve().parents[2]
DEVICE = "cuda"


def execution_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("approved spatial execution requires the measured CUDA device")
    return {**runtime(), "Pillow": importlib_pillow_version(), "python": sys.version,
            "CUDA": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "device_bytes": torch.cuda.get_device_properties(0).total_memory}


def population(plan: dict[str, Any], raw: list[dict[str, Any]], services: dict[str, Any]) -> dict[str, dict]:
    from .regional_study import transaction_for
    entries = {item["key"]: item for item in plan["entries"]}
    identity = services["fas.source_fitting"].identity
    if len(entries) != len(raw) or set(entries) != {identity(row) for row in raw}:
        raise ValueError("exact original spatial population required")
    result = {}
    for row in raw:
        key, entry = identity(row), entries[identity(row)]
        if entry["status"] == "historical_failure":
            if row["detector_status"] == "success" or row["final_k1_action"] != "non_accept":
                raise ValueError("historical failure was recovered")
            result[key] = row
        elif entry["status"] == "shared_geometry_failure":
            from .regional_dino import RegionSupportError
            try:
                points224(transaction_for(row))
            except RegionSupportError as error:
                if str(error) != entry["reason"]:
                    raise ValueError("shared geometry failure changed") from error
            else:
                raise ValueError("shared geometry failure was recovered")
            result[key] = {**row, "detector_status": "failed", "final_k1_action": "non_accept"}
        elif entry["status"] == "usable":
            if (row["detector_status"] != "success"
                    or not np.array_equal(points224(transaction_for(row)), entry["landmarks224"])):
                raise ValueError("spatial geometry/eligibility commitment changed")
            result[key] = row
        else:
            raise ValueError("unknown spatial eligibility state")
    return result


def seal(preparation_path: Path, preparation_sha: str, fas_root: Path, output: Path) -> Path:
    require_private_output(output/"plan.json")
    if output.exists():
        raise FileExistsError("new spatial execution directory required")
    begun = time.monotonic()

    def metadata_guard() -> None:
        if time.monotonic()-begun > 1800:
            raise TimeoutError("execution sealing metadata ceiling")
    value = read_json(preparation_path, preparation_sha)
    if (value["code_sha256"] != code_pins(REPOSITORY) or value["policy"] != POLICY
            or value["settings"] != SETTINGS or value["proposed_limits"] != BUDGET
            or not value["budget_estimate"]["fits_within_proposed_wall_estimate"]
            or value["scientific_execution_authorized"] is not False):
        raise PermissionError("current measured metadata preparation and fitting budget required")
    services, raw, partitions = inputs(fas_root)
    contexts = [item for item in partitions["contexts"] if item["seed"] == SEED]
    if value["contexts"] != contexts:
        raise ValueError("spatial source-only contexts changed")
    indexed = population(value, raw, services)
    from fas.fusion_study import components
    groups = {}
    for context in contexts:
        services["study"].isolation([[indexed[key] for key in context[role]] for role in ROLES])
        originals = [indexed[key] for key in context["evaluation"]]
        groups.update(components([{**row, "group_id": row["subject_id"] or row["group_id"]}
                                  for row in originals]))
    plan = {**value, "study": "spatial_dino_execution_v1", "fas_root": str(fas_root.resolve()),
            "executable_scientific_runner_available": True, "execution_runner_sealed": True, "execution_scope": SCOPE,
            "execution_policy": EXECUTION_POLICY, "budget": BUDGET, "groups": groups,
            "runtime": execution_runtime(), "preparation_path": str(preparation_path.resolve()),
            "preparation_sha256": preparation_sha,
            "files": {**value["files"], str(preparation_path.resolve()): preparation_sha}}
    verify_inputs(plan, metadata_guard)
    path = output/"plan.json"
    write_json(path, plan)
    write_json(output/"preparation.json", preparation_summary(plan, digest(path)))
    return path


def checked_plan(path: Path, sha: str) -> dict[str, Any]:
    require_private_output(path)
    plan = read_json(path, sha)
    if (plan["study"] != "spatial_dino_execution_v1" or plan["execution_scope"] != SCOPE
            or plan["execution_policy"] != EXECUTION_POLICY or plan["budget"] != BUDGET
            or plan["settings"] != SETTINGS or plan["policy"] != POLICY
            or plan["code_sha256"] != code_pins(REPOSITORY) or plan["runtime"] != execution_runtime()
            or plan["scientific_execution_authorized"] is not False
            or plan["executable_scientific_runner_available"] is not True
            or plan["execution_runner_sealed"] is not True):
        raise PermissionError("sealed spatial code/runtime/recipe/scope/budget required")
    return plan


def verify_inputs(plan: dict[str, Any], guard: Callable) -> None:
    source_state(Path(plan["fas_root"]))
    preparation = read_json(Path(plan["preparation_path"]), plan["preparation_sha256"])
    for field in ("settings", "policy", "entries", "contexts", "inventory", "counts", "code_sha256"):
        if plan[field] != preparation[field]:
            raise ValueError(f"execution/preparation binding changed: {field}")
    expected = {**preparation["files"], plan["preparation_path"]: plan["preparation_sha256"]}
    if plan["files"] != expected:
        raise ValueError("complete spatial file bindings required")
    for filename, sha in plan["files"].items():
        guard()
        if digest(Path(filename)) != sha:
            raise ValueError(f"spatial bound input changed: {Path(filename).name}")
    services, raw, partitions = inputs(Path(plan["fas_root"]))
    if plan["contexts"] != [c for c in partitions["contexts"] if c["seed"] == SEED]:
        raise ValueError("exact four-target context inventory required")
    indexed = population(plan, raw, services)
    from fas.fusion_study import components
    expected_groups = {}
    for context in plan["contexts"]:
        services["study"].isolation([[indexed[key] for key in context[role]] for role in ROLES])
        originals = [indexed[key] for key in context["evaluation"]]
        expected_groups.update(components([{**row, "group_id": row["subject_id"] or row["group_id"]}
                                          for row in originals]))
        for arm in ARMS:
            counts = {role: sum(indexed[key]["detector_status"] == "success" for key in context[role])
                      for role in ROLES}
            record = {"target": context["target"], "arm": arm, "seed": SEED,
                      **{f"{role}_usable": count for role, count in counts.items()},
                      "maximum_optimizer_steps": ((counts["fitting"]+31)//32)*SETTINGS["epochs"]}
            if record not in plan["inventory"]:
                raise ValueError("exact twenty-fit usable inventory required")
    if len(plan["inventory"]) != 20:
        raise ValueError("exact twenty-fit inventory length required")
    if plan["groups"] != expected_groups:
        raise ValueError("exact subject/group/content bootstrap components required")


def clock(approval: dict[str, Any]) -> None:
    now = time.monotonic()
    if (approval["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            or not 0 <= now-approval["approved_monotonic"] < BUDGET["max_seconds"]):
        raise TimeoutError("original nonresettable spatial approval clock required")


def require_approval(path: Path, sha: str, approval_path: Path) -> tuple[dict, dict]:
    plan = checked_plan(path, sha)
    approval = read_json(approval_path)
    if (approval["answer"] != ANSWER or approval["plan_sha256"] != sha
            or approval["budget"] != BUDGET or approval["scope"] != SCOPE
            or approval["execution_policy"] != EXECUTION_POLICY):
        raise PermissionError("exact spatial plan owner approval required")
    clock(approval)
    return plan, approval


def approve(path: Path, sha: str, answer: str) -> Path:
    plan = checked_plan(path, sha)
    if answer != ANSWER or any((path.parent/name).exists() for name in
                              ("approval.json", "attempt.json", "started.json", "batch.log")):
        raise PermissionError("explicit exact-plan answer and unapproved/unattempted plan required")
    begun = time.monotonic()

    def guard() -> None:
        if time.monotonic()-begun > 1800:
            raise TimeoutError("approval metadata verification cap")

    verify_inputs(plan, guard)
    output = path.parent/"approval.json"
    write_json(output, {"answer": answer, "plan_sha256": sha, "budget": BUDGET, "scope": SCOPE,
                        "execution_policy": EXECUTION_POLICY, "approved_monotonic": begun,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
    return output


def loaders(plan: dict[str, Any], guard: Callable) -> tuple[Callable, Callable]:
    entries = {entry["key"]: entry for entry in plan["entries"]}

    def rgb(row: dict[str, Any]) -> np.ndarray:
        guard()
        crop = row["crops"]["224"]
        payload = Path(crop["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != crop["sha256"]:
            raise ValueError("spatial RGB224 bytes changed before decode")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError("original RGB224 decode required, no fallback")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @lru_cache(maxsize=256)
    def cached_prior(key: str) -> np.ndarray:
        guard()
        entry = entries[key]
        if entry["status"] != "usable":
            raise ValueError("prior requested for retained shared failure")
        return prior_image(np.asarray(entry["landmarks224"], dtype=float))

    def prior(row: dict[str, Any]) -> np.ndarray:
        return cached_prior(f"{row['dataset']}/{row['video_id']}")

    return rgb, prior


def analyze(target: str, predictions: dict, policies: dict, originals: list[dict],
            services: dict, guard: Callable, groups: dict) -> tuple[list, list]:
    study = services["study"]
    cases = []
    for arm in ARMS:
        positions, labels = study.score_inputs(predictions[arm], originals)
        for alpha in study.ALPHAS:
            point = study.metrics(predictions[arm], originals, policies[arm][str(alpha)]["threshold"])
            point["FA_within_constraint"] = point["accepted_attack"] <= (
                int(point["original_attack"])*alpha.numerator//alpha.denominator)
            point["utility_feasible"] = point["FA_within_constraint"] and point["BFA"] >= .5
            oracle = services["oracle316"].oracle(predictions[arm], labels[positions],
                int(labels.sum()), int((1-labels).sum()), alpha=alpha)
            cases.append({"target": target, "seed": SEED, "arm": arm, "alpha": str(alpha),
                          "policy": point, "oracle": oracle})
    intervals = paired_intervals(predictions, originals, policies, services, guard,
        arms=ARMS, contrasts=CONTRASTS, repetitions=EXECUTION_POLICY["bootstrap_repetitions"], group_map=groups)
    return cases, [{"target": target, "seed": SEED, **record} for record in intervals]


def verify_paired_traces(paths: list[Path], steps: list[int], guard: Callable) -> None:
    from contextlib import ExitStack
    from itertools import zip_longest
    counts = [0]*len(paths)
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open()) for path in paths]
        for lines in zip_longest(*streams):
            guard()
            records = []
            for i, line in enumerate(lines):
                if line is not None:
                    counts[i] += 1
                    records.append(json.loads(line))
            if any(record != records[0] for record in records[1:]):
                raise ValueError("five-arm addressed batch/geometry pairing mismatch")
    if counts != steps:
        raise ValueError("paired trace optimizer-step inventory mismatch")


def worker(path: Path, sha: str, approval_path: Path) -> None:
    plan, approval = require_approval(path, sha, approval_path)
    root = path.parent
    attempt = read_json(root/"attempt.json")
    if attempt["supervisor_pid"] != os.getppid() or attempt["plan_sha256"] != sha:
        raise PermissionError("spatial worker requires original recorded supervisor")
    supervisor = os.getppid()
    write_json(root/"started.json", {"worker_pid": os.getpid(), "plan_sha256": sha,
                                    "approval_sha256": digest(approval_path)})

    def guard() -> None:
        clock(approval)
        if time.monotonic() >= approval["approved_monotonic"]+BUDGET["max_seconds"]-60:
            raise TimeoutError("spatial computation reached shared closure reserve")
        if os.getppid() != supervisor:
            raise RuntimeError("spatial worker lost supervisor")

    verify_inputs(plan, guard)
    torch.cuda.set_per_process_memory_fraction(BUDGET["max_gpu_bytes"]/
                                               torch.cuda.get_device_properties(0).total_memory)
    services, raw_rows, _ = inputs(Path(plan["fas_root"]))
    indexed = population(plan, raw_rows, services)
    load_rgb, load_prior = loaders(plan, guard)
    source = source_modules(Path(plan["fas_root"]))
    fitting, study = services["fas.source_fitting"], services["study"]
    cases, intervals, artifacts = [], [], {}
    initial_reference, initial_fusion = None, None
    for context in plan["contexts"]:
        target = context["target"]
        pools = {role: [indexed[key] for key in context[role]
                        if indexed[key]["detector_status"] == "success"] for role in ROLES}
        keys = {role: [fitting.identity(row) for row in pool] for role, pool in pools.items()}
        predictions, policies, models, step_counts, traces = {}, {}, {}, [], []
        for arm in ARMS:
            guard()
            folder = root/"models"/f"{target}-{arm}"
            write_json(folder/"started.json", {"target": target, "arm": arm, "seed": SEED, "plan_sha256": sha})
            print("SPATIAL_TRAIN_START", target, arm, flush=True)
            source.model.configure(SEED)
            model = SpatialReference(source.model.Reference(), arm).to(DEVICE)
            reference_sha = tensor_digest(model.reference.state_dict())
            fusion_sha = None if model.fusion is None else tensor_digest(model.fusion.state_dict())
            if initial_reference is not None and initial_reference != reference_sha:
                raise ValueError("all spatial fits require identical initial reference state")
            initial_reference = reference_sha
            if fusion_sha is not None:
                if initial_fusion is not None and initial_fusion != fusion_sha:
                    raise ValueError("all local arms require identical initial fusion tensors")
                initial_fusion = fusion_sha
            trace = folder/"batches.jsonl"
            state = fit(model, pools["fitting"], pools["validation"], load_rgb, load_prior,
                        source, SEED, DEVICE, guard, trace)
            validate_state(state, len(pools["fitting"]), arm)
            checkpoint = folder/"model.pth"
            with checkpoint.open("xb") as stream:
                torch.save(state, stream)
            raw = {role: predict(model, pools[role], load_rgb, load_prior, source, DEVICE, guard)
                   for role in ("validation", "calibration", "evaluation")}
            write_json(folder/"calibrator-started.json", {"plan_sha256": sha})
            calibrator = fitting.fit_calibrator(raw["calibration"],
                np.asarray([row["binary_label"] == "attack" for row in pools["calibration"]], dtype=int),
                [row["dataset"] for row in pools["calibration"]], guard)
            probabilities = {role: fitting.calibrate(calibrator, raw[role])
                             for role in ("calibration", "evaluation")}
            policies[arm] = {str(alpha): study.threshold(probabilities["calibration"], pools["calibration"], alpha)
                             for alpha in study.ALPHAS}
            predictions[arm] = probabilities["evaluation"]
            models[arm] = {"checkpoint_sha256": digest(checkpoint), "trace_sha256": digest(trace),
                "initial_reference_sha256": reference_sha, "initial_fusion_sha256": fusion_sha,
                "selected_epoch": state["selected_epoch"], "optimizer_steps": state["optimizer_steps"],
                "raw_logits": {role: values.tolist() for role, values in raw.items()},
                "calibrator": calibrator,
                "probabilities": {role: values.tolist() for role, values in probabilities.items()}}
            traces.append(trace)
            step_counts.append(state["optimizer_steps"])
            print("SPATIAL_TRAIN_COMPLETE", target, arm, state["selected_epoch"], flush=True)
            del model, state, raw
            gc.collect()
            torch.cuda.empty_cache()
        verify_paired_traces(traces, step_counts, guard)
        record = {"target": target, "keys": keys, "models": models, "policies": policies,
                  "prediction": {arm: values.tolist() for arm, values in predictions.items()}}
        write_json(root/f"context-{target}.json", record)
        artifacts[target] = digest(root/f"context-{target}.json")
        originals = [indexed[key] for key in context["evaluation"]]
        points, uncertainty = analyze(target, predictions, policies, originals, services, guard,
                                     {key: plan["groups"][key] for key in context["evaluation"]})
        cases.extend(points)
        intervals.extend(uncertainty)
    if len(cases) != SCOPE["operating_cases"] or len(intervals) != SCOPE["conditional_intervals"]:
        raise ValueError("complete five-arm/four-target output required")
    write_json(root/"result.json", {"plan_sha256": sha, "artifacts": artifacts, "cases": cases,
                                    "conditional_intervals": intervals})
    del load_rgb, load_prior
    gc.collect()
    torch.cuda.empty_cache()
    guard()
    print("SPATIAL_VERIFICATION_START", flush=True)
    subprocess.run([sys.executable, str(REPOSITORY/"scripts/spatial_dino.py"), "_verify",
        "--plan", str(path), "--plan-sha256", sha, "--approval", str(approval_path)],
        check=True, timeout=max(.01, approval["approved_monotonic"]+BUDGET["max_seconds"]-time.monotonic()))


def preparation_summary(plan: dict[str, Any], sha: str) -> dict[str, Any]:
    return {"plan_sha256": sha, "scope": SCOPE, "execution_policy": EXECUTION_POLICY,
            "budget": BUDGET, "settings": SETTINGS, "counts": plan["counts"],
            "inventory": plan["inventory"], "bound_file_count": len(plan["files"]),
            "budget_estimate": plan["budget_estimate"], "scientific_execution_authorized": False,
            "execution_runner_sealed": True, "PAD_crop_decodes": 0, "scientific_training_starts": 0}


def supervise(path: Path, sha: str, approval_path: Path) -> dict[str, Any]:
    from .adapter_pilot import device_memory_bytes
    plan, approval = require_approval(path, sha, approval_path)
    root = path.parent
    if any((root/name).exists() for name in ("attempt.json", "started.json", "batch.log", "supervision.json")):
        raise FileExistsError("one spatial attempt only; no retry/resume/clock reset")
    if device_memory_bytes() >= BUDGET["max_gpu_bytes"]:
        raise MemoryError("shared device already exceeds the conservative GPU budget")
    remaining = approval["approved_monotonic"]+BUDGET["max_seconds"]-30-time.monotonic()
    if remaining <= 0:
        raise TimeoutError("spatial approval has no supervised closure reserve remaining")
    write_json(root/"attempt.json", {"plan_sha256": sha, "supervisor_pid": os.getpid(),
                                    "approval_sha256": digest(approval_path)})
    result = run_bounded(
        [sys.executable, str(REPOSITORY/"scripts/spatial_dino.py"), "_worker", "--plan", str(path),
         "--plan-sha256", sha, "--approval", str(approval_path)], cwd=REPOSITORY,
        owned_root=root, log=root/"batch.log",
        max_seconds=remaining,
        max_rss_bytes=BUDGET["max_rss_bytes"], max_disk_bytes=BUDGET["max_output_bytes"],
        max_gpu_bytes=BUDGET["max_gpu_bytes"], gpu_memory_bytes=device_memory_bytes, rss_root_pid=os.getpid())
    write_json(root/"supervision.json", {"plan_sha256": sha, **result})
    if not result["completed"]:
        write_json(root/"terminal.json", {"completed": False, "plan_sha256": sha,
            "supervision_sha256": digest(root/"supervision.json"), "scientific_success_claimed": False})
        raise RuntimeError("spatial attempt failed; retained evidence, no automatic retry")
    verification = read_json(root/"verification.json")
    if (verification["completed"] is not True or verification["plan_sha256"] != sha
            or verification["result_sha256"] != digest(root/"result.json")):
        raise ValueError("separate complete spatial checkpoint reconstruction required")
    clock(approval)
    summary = completed_summary(root, sha, verification, result)
    write_json(root/"aggregate.json", summary)
    clock(approval)
    if tree_bytes(root)+8192 > BUDGET["max_output_bytes"]:
        raise MemoryError("spatial closure exceeds owned output budget")
    write_json(root/"completion.json", {"completed": True, "plan_sha256": sha,
        "aggregate_sha256": digest(root/"aggregate.json"), "verification_sha256": digest(root/"verification.json"),
        "supervision_sha256": digest(root/"supervision.json")})
    write_json(root/"terminal.json", {"completed": True, "plan_sha256": sha,
                                     "completion_sha256": digest(root/"completion.json")})
    return summary


def completed_summary(root: Path, sha: str, verification: dict, supervision: dict) -> dict:
    result = read_json(root/"result.json")
    if result["plan_sha256"] != sha or verification["completed"] is not True or not supervision["completed"]:
        raise ValueError("successful fully bound spatial result required")
    if (verification["checkpoint_replays"] != 20 or verification["new_fits"] != 0
            or len(result["cases"]) != SCOPE["operating_cases"]
            or len(result["conditional_intervals"]) != SCOPE["conditional_intervals"]):
        raise ValueError("complete spatial reconstruction and output inventories required")
    macro = []
    for arm in ARMS:
        for alpha in ("1/20", "1/100", "1/10"):
            selected = [case["policy"] for case in result["cases"] if case["arm"] == arm and case["alpha"] == alpha]
            if len(selected) != 4:
                raise ValueError("all targets required in spatial summary")
            macro.append({"arm": arm, "alpha": alpha, "equal_target_macro": {
                metric: float(np.mean([point[metric] for point in selected]))
                for metric in ("FA", "BFA", "AUROC_usable")},
                "FA_safe_targets": sum(point["FA_within_constraint"] for point in selected),
                "joint_utility_targets": sum(point["utility_feasible"] for point in selected)})
    return {"completed": True, "plan_sha256": sha, "cases": result["cases"],
            "conditional_intervals": result["conditional_intervals"], "macro": macro,
            "verification": verification, "supervision": supervision, "scope": SCOPE,
            "development_targets_already_seen": True, "SiW_subject_independence_verified": False}
