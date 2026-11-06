"""Fixed two-crop contract for the shared spatial training and replay engine."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from .io import digest, object_digest, read_json, require_private_output, write_json
from .regional_study import inputs, source_state
from .spatial_crop_preparation import ARMS, CROPS, POLICY, footprint_check, inventory
from .spatial_diagnosis import crop_candidates
from .spatial_dino import SEED, SETTINGS as ENGINE_SETTINGS, prior_image
from .spatial_preparation import PROBE_CAPS, code_pins, estimate_budget

PREPARATION_SHA = "b82353431baa0fa08e467b8d785fd48abb11c1d54518ca1af52a5634186ca71e"
SUMMARY_SHA = "66d4521f9166cde694b675eb9105d24febcd0e988921e44daa8435c67254e6e8"
VERIFICATION_SHA = "f67bd0a45d553cbce668c111294a1f84d467c1b90f722ee0033fdddc589ba024"
SYSTEMS = tuple(f"{crop}_{arm}" for crop in CROPS for arm in ARMS)
VIEWS = ("original", "common")
SETTINGS = {**ENGINE_SETTINGS, "geometry": "paired_crop_specific_RGB224_pixel_center_landmarks"}
CONTRASTS = tuple(
    (f"{crop}_{left}_minus_{right}", f"{crop}_{left}", f"{crop}_{right}")
    for crop in CROPS for left, right in (
        ("anatomical", "content"), ("anatomical", "reference"), ("content", "reference"))
) + tuple((f"square_minus_legacy_{arm}", f"square_{arm}", f"legacy_{arm}") for arm in ARMS)
BUDGET = {"max_seconds": 120*3600, "max_rss_bytes": 8*1024**3,
          "max_gpu_bytes": 14*1024**3, "max_output_bytes": 16*1024**3}
SCOPE = {
    "targets": 4, "seed": SEED, "backbone_fits": 24, "new_backbone_fits": 24,
    "calibrator_fits": 24, "reused_backbone_fits": 0, "operating_cases": 144,
    "conditional_intervals": 360, "views": list(VIEWS), "systems": list(SYSTEMS),
    "PAD_crop_decode": True, "new_native_video_decode": False, "new_detector": False,
    "generic_pretrained_initialization_only": True, "new_download": False,
    "new_environment": False, "automatic_retry": False, "automatic_expansion": False,
}
EXECUTION_POLICY = {
    "primary_candidate": "square_anatomical", "reference": "legacy_reference",
    "paired_contrasts": [list(item) for item in CONTRASTS],
    "bootstrap_repetitions": 2000, "bootstrap_seed": 20261028,
    "selection": ENGINE_SETTINGS["selection"], "target_model_selection": False,
    "oracle_diagnostic_only": True, "failure": "fail_closed_retained_receipts_no_retry_or_resume",
    "shared_clock": "approval_through_input_checks_training_replay_analysis_and_closure",
    "verification": "separate_GPU_checkpoint_replay_no_new_fit_independent_counts_and_bootstrap",
    "crop_pairing": "same_original_order_RNG_transforms_labels_and_PDA_donors_not_equal_RGB_or_priors",
    "checkpoint_inputs": "explicit_target_crop_arm_preparation_mask_and_role_key_commitment",
    "views": "one_calibrator_per_fit_shared_threshold_both_denominators_no_refitting",
}
ANSWER = "approve_exact_spatial_crop_24_fits_120h_no_retry"


def split_system(system: str) -> tuple[str, str]:
    if system not in SYSTEMS:
        raise ValueError("explicit fixed crop/model system required")
    return next((crop, arm) for crop in CROPS for arm in ARMS if system == f"{crop}_{arm}")


def input_binding(plan: dict, target: str, system: str) -> dict:
    crop, arm = split_system(system)
    context = next(context for context in plan["contexts"] if context["target"] == target)
    entries = {entry["key"]: entry for entry in plan["entries"]}
    keys = {role: [key for key in context[role] if entries[key]["status"] == "usable"]
            for role in ("fitting", "validation", "calibration", "evaluation")}
    return {"schema": "spatial_crop_inputs_v1", "target": target, "system": system,
            "crop": crop, "model_arm": arm, "preparation_sha256": plan["preparation_sha256"],
            "common_eligibility_sha256": plan["common_eligibility_sha256"],
            "role_keys_sha256": object_digest(keys)}


def views(originals: list[dict], groups: dict) -> list[tuple[str, list[dict], dict]]:
    common = [row for row in originals if row["detector_status"] == "success"]
    keys = {f"{row['dataset']}/{row['video_id']}" for row in common}
    return [("original", originals, groups),
            ("common", common, {key: value for key, value in groups.items() if key in keys})]


def population(plan: dict, raw: list[dict], services: dict) -> dict:
    from .spatial_study import population as historical_population
    origin = read_json(Path(plan["crop_study"]["origin_plan_path"]),
                       plan["crop_study"]["origin_plan_sha256"])
    indexed = historical_population(origin, raw, services)
    entries = {entry["key"]: entry for entry in plan["entries"]}
    if len(entries) != len(raw) or set(entries) != set(indexed):
        raise ValueError("exact paired-crop original population required")
    for key, row in indexed.items():
        if entries[key]["status"] == "failed":
            indexed[key] = {**row, "detector_status": "failed", "final_k1_action": "non_accept"}
        elif entries[key]["status"] != "usable" or row["detector_status"] != "success":
            raise ValueError("unknown paired-crop status or recovered failure")
    return indexed


def loaders(plan: dict, guard: Callable, system: str) -> tuple[Callable, Callable]:
    crop, _ = split_system(system)
    entries = {entry["key"]: entry for entry in plan["entries"]}

    def entry_for(row: dict) -> dict:
        entry = entries[f"{row['dataset']}/{row['video_id']}"]
        if entry["status"] != "usable":
            raise ValueError("crop requested for retained shared failure")
        return entry

    def rgb(row: dict) -> np.ndarray:
        guard()
        asset = entry_for(row)["crops"][crop]
        payload = Path(asset["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != asset["sha256"]:
            raise ValueError("crop-specific RGB224 payload changed before decode")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError("explicit crop-specific RGB224 decode required, no fallback")
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if crop == "square" and hashlib.sha256(result.tobytes()).hexdigest() != asset["RGB_sha256"]:
            raise ValueError("square RGB224 decoded pixels differ from preparation")
        return result

    @lru_cache(maxsize=256)
    def cached_prior(key: str) -> np.ndarray:
        guard()
        entry = entries[key]
        if entry["status"] != "usable":
            raise ValueError("prior requested for retained shared failure")
        return prior_image(np.asarray(entry["landmarks224"][crop], dtype=float))

    def prior(row: dict) -> np.ndarray:
        entry_for(row)
        return cached_prior(f"{row['dataset']}/{row['video_id']}")

    return rgb, prior


def verify_traces(paths: list[Path], steps: list[int], guard: Callable) -> None:
    from .spatial_study import verify_paired_traces
    if len(paths) != 6 or len(steps) != 6:
        raise ValueError("six crop/arm trace inventories required")
    for offset in (0, 3):
        verify_paired_traces(paths[offset:offset+3], steps[offset:offset+3], guard)
    verify_paired_traces(paths, steps, guard, omit=("batch_sha256", "spatial_prior_sha256"))


def verify_inputs(plan: dict, guard: Callable) -> None:
    from .regional_study import transaction_for
    source_state(Path(plan["fas_root"]))
    preparation = read_json(Path(plan["preparation_path"]), PREPARATION_SHA)
    summary = read_json(Path(plan["crop_study"]["summary_path"]), SUMMARY_SHA)
    verification = read_json(Path(plan["crop_study"]["verification_path"]), VERIFICATION_SHA)
    if (verification["completed"] is not True or verification["preparation_sha256"] != PREPARATION_SHA
            or verification["summary_sha256"] != SUMMARY_SHA or summary["common_usable"] != 6231):
        raise ValueError("complete fixed-cohort independent crop preparation required")
    for field in ("entries", "contexts", "groups", "jobs", "roles", "common_eligibility_sha256"):
        if plan[field] != preparation[field]:
            raise ValueError(f"paired-crop input commitment drift: {field}")
    expected_files = {**preparation["files"], plan["preparation_path"]: PREPARATION_SHA,
                      plan["crop_study"]["summary_path"]: SUMMARY_SHA,
                      plan["crop_study"]["verification_path"]: VERIFICATION_SHA,
                      plan["crop_study"]["probe_path"]: plan["crop_study"]["probe_sha256"],
                      plan["crop_study"]["probe_supervision_path"]: plan["crop_study"]["probe_supervision_sha256"]}
    if plan["files"] != expected_files:
        raise ValueError("complete paired-crop file bindings required")
    for filename, sha in plan["files"].items():
        guard()
        if digest(Path(filename)) != sha:
            raise ValueError(f"paired-crop bound input changed: {Path(filename).name}")
    services, raw, partitions = inputs(Path(plan["fas_root"]))
    if plan["contexts"] != [c for c in partitions["contexts"] if c["seed"] == SEED]:
        raise ValueError("original source-only role assignments changed")
    indexed = population(plan, raw, services)
    jobs, roles = inventory(plan["contexts"], indexed, plan["entries"], services)
    if plan["jobs"] != jobs or plan["inventory"] != jobs or plan["roles"] != roles:
        raise ValueError("exact matched24-fit strata/key inventory required")
    by_original = {entry["key"]: entry for entry in
                   read_json(Path(plan["crop_study"]["origin_plan_path"]),
                             plan["crop_study"]["origin_plan_sha256"])["entries"]}
    for entry in plan["entries"]:
        guard()
        if entry["status"] != "usable":
            continue
        transaction = transaction_for(indexed[entry["key"]])
        candidate = crop_candidates(transaction)["candidates"]["expanded_legacy_square"]
        if not candidate["strict_pair_eligible"]:
            raise ValueError("frozen crop eligibility no longer has real-pixel support")
        footprint_check(transaction, candidate)
        if (entry["landmarks224"]["legacy"] != by_original[entry["key"]]["landmarks224"]
                or entry["landmarks224"]["square"] != candidate["landmarks224"]
                or entry["source_to_square_matrix"] != candidate["source_to_RGB224_matrix"]
                or entry["crops"]["legacy"] != indexed[entry["key"]]["crops"]["224"]):
            raise ValueError("crop-specific prior or legacy RGB binding drift")
    probe = read_json(Path(plan["crop_study"]["probe_path"]), plan["crop_study"]["probe_sha256"])
    supervisor = read_json(Path(plan["crop_study"]["probe_supervision_path"]),
                           plan["crop_study"]["probe_supervision_sha256"])
    check_probe(probe, supervisor)
    if probe["code_sha256"] != plan["code_sha256"]:
        raise ValueError("resource probe must measure current engine/controller code")
    if plan["budget_estimate"] != budget_estimate(jobs, probe):
        raise ValueError("paired-crop synthetic budget estimate drift")


def budget_estimate(jobs: list[dict], probe: dict) -> dict:
    converted = [{"arm": job["arm"], "maximum_optimizer_steps": job["maximum_optimizer_steps"],
                  "validation_usable": job["usable"]["validation"]} for job in jobs]
    estimate = estimate_budget(converted, probe)
    estimate["proposed_wall_cap_seconds"] = BUDGET["max_seconds"]
    estimate["fits_within_proposed_wall_estimate"] = (
        estimate["planning_seconds_with_3x_factor_plus_6h_reserve"] <= BUDGET["max_seconds"])
    estimate["crop_IO_measured"] = False
    estimate["engineering_probe_is_shared_five_arm_superset"] = True
    if estimate["estimated_checkpoint_bytes"]+2*1024**3 > BUDGET["max_output_bytes"]:
        raise MemoryError("24 checkpoints plus2GiB trace/analysis reserve exceed output budget")
    return estimate


def check_probe(probe: dict, supervisor: dict) -> None:
    if (supervisor["completed"] is not True or supervisor["guard_failure"] is not None
            or supervisor["returncode"] != 0
            or supervisor["caps"] != {**PROBE_CAPS, "max_gpu_bytes": BUDGET["max_gpu_bytes"]}
            or probe["caps"] != PROBE_CAPS or probe["synthetic_optimizer_steps"] != 15
            or len(probe["results"]) != 5):
        raise ValueError("successful bounded synthetic superset probe and supervision required")
    for result in probe["results"]:
        times = result["measured_step_seconds"]
        if (len(times) != 2 or not np.isfinite(times).all() or min(times) <= 0
                or not result["same_process_replay_exact"] or not result["independent_replay_exact"]):
            raise ValueError("positive synthetic step measurements and exact checkpoint replays required")


def seal(preparation_path: Path, verification_path: Path, probe_path: Path, output: Path) -> Path:
    from .spatial_study import execution_runtime
    require_private_output(output/"plan.json")
    if output.exists():
        raise FileExistsError("new immutable paired-crop execution directory required")
    start = time.monotonic()

    def guard() -> None:
        if time.monotonic()-start >= 1800:
            raise TimeoutError("paired-crop metadata sealing cap")

    preparation = read_json(preparation_path, PREPARATION_SHA)
    summary_path = preparation_path.parent/"summary.json"
    read_json(summary_path, SUMMARY_SHA)
    read_json(verification_path, VERIFICATION_SHA)
    probe = read_json(probe_path)
    probe_supervisor_path = probe_path.parent/"supervision.json"
    check_probe(probe, read_json(probe_supervisor_path))
    if preparation["settings"] != SETTINGS or preparation["policy"] != POLICY:
        raise ValueError("fixed approved crop geometry and recipe required")
    estimate = budget_estimate(preparation["jobs"], probe)
    if not estimate["fits_within_proposed_wall_estimate"]:
        raise TimeoutError("synthetic24-fit planning estimate exceeds proposed120h budget")
    origin_paths = [Path(name) for name, sha in preparation["files"].items()
                    if sha == preparation["origin"]["plan.json"] and Path(name).name == "plan.json"]
    if len(origin_paths) != 1:
        raise ValueError("unique completed original spatial plan binding required")
    plan = {**preparation, "study": "spatial_crop_execution_v1",
            "code_sha256": code_pins(Path(__file__).resolve().parents[2]),
            "runtime": execution_runtime(), "budget": BUDGET, "execution_scope": SCOPE,
            "execution_policy": EXECUTION_POLICY, "settings": SETTINGS, "policy": POLICY,
            "execution_runner_sealed": True, "executable_scientific_runner_available": True,
            "scientific_execution_authorized": False,
            "preparation_path": str(preparation_path.resolve()), "preparation_sha256": PREPARATION_SHA,
            "inventory": preparation["jobs"],
            "counts": {"originals": 7505, "usable": 6231, "shared_failures": 1274},
            "budget_estimate": estimate,
            "crop_study": {"origin_plan_path": str(origin_paths[0]),
                "origin_plan_sha256": preparation["origin"]["plan.json"],
                "summary_path": str(summary_path.resolve()), "verification_path": str(verification_path.resolve()),
                "probe_path": str(probe_path.resolve()), "probe_sha256": digest(probe_path),
                "probe_supervision_path": str(probe_supervisor_path.resolve()),
                "probe_supervision_sha256": digest(probe_supervisor_path)},
            "files": {**preparation["files"], str(preparation_path.resolve()): PREPARATION_SHA,
                str(summary_path.resolve()): SUMMARY_SHA, str(verification_path.resolve()): VERIFICATION_SHA,
                str(probe_path.resolve()): digest(probe_path),
                str(probe_supervisor_path.resolve()): digest(probe_supervisor_path)}}
    verify_inputs(plan, guard)
    path = output/"plan.json"
    write_json(path, plan)
    from .spatial_study import preparation_summary
    write_json(output/"preparation.json", preparation_summary(plan, digest(path)))
    return path
