"""Immutable, approval-gated six-arm regional DINO study on accepted FAS inputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import gc
import hashlib
import importlib
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

from .dino_comparison import DINO_REVISION, PARTITIONS_SHA, SEEDS, WEIGHT_SHAS
from .gpu_probe import tensor_digest
from .io import digest, read_json, require_private_output, write_json
from .regional_dino import (
    ARMS, CONTRASTS, DIMENSIONS, HEAD_SETTINGS, REGIONS, REPRESENTATIONS, RegionSupportError, crop_landmarks,
    fit_group_dro, pool_tokens, region_weights,
)

SOURCE_REVISION = "707e3ddf6103835c55e937cbbd84081d91f188c7"
BUDGET = {"max_seconds": 4*3600, "max_rss_bytes": 8*1024**3,
          "max_gpu_bytes": 8*1024**3, "max_output_bytes": 4*1024**3}
SCOPE = {"new_video_decodes": 0, "new_detectors": 0, "encoder_fitting": False,
         "new_download": False, "automatic_retry": False, "contexts": 12,
         "head_fits": 72, "calibrator_fits": 72,
         "source_fitting_validation_calibration_pixels": True, "target_test_pixels": True,
         "single_accepted_middle_frame": True, "legacy_RGB518_crop_unchanged": True,
         "invalid_regions_shared_technical_failure_all_arms_originals_retained": True,
         "SiW_subject_independence_verified": False, "untouched_confirmation": False}
POLICY = {
    "regions": list(REGIONS), "radius_reference": "fixed_MAT_eye_distance_ellipses_not_reconstruction",
    "patch_rule": "patch_center_inside_full_ellipse_no_clipping_or_empty_fallback",
    "control": "nearest_same_patch_count_to_fixed_quarter_image_anchors_stable_ties",
    "global": "normalized_CLS_concat_mean_normalized_image_patch_tokens_excluding_registers",
    "regional": "global_concat_four_region_means_no_feature_standardization",
    "validation": "source_only_equal_source_equal_class_BCE_earliest_strict_best_no_refit",
    "threshold": "minimum_source_safe_attack_order_statistic_successful_calibration_budget",
    "accept": "probability_strictly_less_than_threshold_ties_rejected",
    "primary_alpha": "1/20", "sensitivity_alphas": ["1/100", "1/10"],
    "bootstrap_repetitions": 2000, "bootstrap_seed": 20261028,
    "paired_contrasts": [list(contrast) for contrast in CONTRASTS],
    "primary_candidate": "anatomical__group_dro", "reference": "global__erm",
    "target_model_selection": False, "oracle_is_diagnostic_only": True}


def source_services(root: Path) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root/"src"), str(root/"scripts"), str(root/"scripts/diagnostics"),
                   str(root/"scripts/paper321")]
    modules = {}
    for name, relative in (
        ("prepare_paper319", "scripts/diagnostics/prepare_paper319.py"),
        ("study", "scripts/paper321/study.py"),
        ("oracle316", "scripts/diagnostics/oracle316.py"),
        ("paper321.runner", "scripts/paper321/runner.py"),
        ("fas.source_fitting", "src/fas/source_fitting.py"),
        ("fas.frozen_features", "src/fas/frozen_features.py"),
    ):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (root/relative).resolve():
            raise ValueError(f"unbound FAS service import: {name}")
        modules[name] = module
    return modules


def source_state(root: Path) -> None:
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    if revision != SOURCE_REVISION or dirty:
        raise ValueError("clean pinned read-only FAS source required")


def runtime() -> dict[str, str]:
    return {name: version(name) for name in ("numpy", "torch", "opencv-python-headless", "scipy", "threadpoolctl")}


def implementation() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [root/"scripts/regional_dino.py", *sorted((root/"src").rglob("*.py"))]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def transaction_for(row: dict[str, Any]) -> dict[str, Any]:
    transaction = read_json(Path(row["record_path"]), row["record_sha256"])
    if (transaction["video_id"] != row["video_id"] or transaction["detector_status"] != "success"
            or transaction["selected_frame"] != row["selected_frame"]
            or transaction["binary_label"] != row["binary_label"]):
        raise ValueError("accepted crop/landmark transaction join changed")
    return transaction


def inputs(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    source_state(root)
    services = source_services(root)
    preparation = services["prepare_paper319"]
    manifest = read_json(preparation.MANIFEST, preparation.MANIFEST_SHA)
    rows = preparation.active_rows(manifest["rows"])
    partitions = read_json(preparation.OUTPUT/"partitions.json", PARTITIONS_SHA)
    rebuilt, _ = preparation.build_partitions(manifest["rows"])
    if partitions != rebuilt:
        raise ValueError("exact accepted official-partition reconstruction required")
    services["paper321.runner"].validate_partitions(rows, partitions)
    if (len(rows) != 7505 or sum(row["detector_status"] == "success" for row in rows) != 7474
            or len(partitions["contexts"]) != 12):
        raise ValueError("fixed completed four-dataset population required")
    return services, rows, partitions


def regional_population(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    result, failures = [], {}
    for row in rows:
        if row["detector_status"] != "success":
            result.append(row)
            continue
        points = crop_landmarks(transaction_for(row))
        try:
            region_weights(points)
        except RegionSupportError as error:
            key = f"{row['dataset']}/{row['video_id']}"
            failures[key] = str(error)
            result.append({**row, "detector_status": "failed", "original_detector_status": "success",
                           "failure_kind": str(error), "model_input_eligible": False,
                           "score_fit_eligible": False, "final_k1_action": "non_accept"})
        else:
            result.append(row)
    return result, failures


def build_plan(fas_root: Path, output: Path) -> Path:
    require_private_output(output/"plan.json")
    if output.exists():
        raise FileExistsError("new regional study root required")
    begun = time.monotonic()
    services, original_rows, partitions = inputs(fas_root)
    rows, failures = regional_population(original_rows)
    preparation = services["prepare_paper319"]
    weights = Path.home()/".cache/fas/source-pilot295-v1/checkpoints/dino_reg.pth"
    upstream = Path.home()/".cache/fas/upstream/dinov2-7764ea0"
    files = {str(preparation.MANIFEST): preparation.MANIFEST_SHA,
             str(preparation.OUTPUT/"partitions.json"): PARTITIONS_SHA, str(weights): WEIGHT_SHAS[0]}
    counts: Counter[str] = Counter()
    minima = {name: 1369 for name in REGIONS}
    for row in original_rows:
        if row["detector_status"] == "success":
            files[row["record_path"]] = row["record_sha256"]
            files[row["crops"]["518"]["path"]] = row["crops"]["518"]["sha256"]
    for row in rows:
        if time.monotonic()-begun > 1800:
            raise TimeoutError("metadata preflight 30-minute cap")
        if row["detector_status"] != "success":
            if row["final_k1_action"] != "non_accept":
                raise ValueError("original technical failures must remain nonacceptances")
            continue
        transaction = transaction_for(row)
        weights_by_region = region_weights(crop_landmarks(transaction))
        for index, name in enumerate(REGIONS):
            minima[name] = min(minima[name], int(np.count_nonzero(weights_by_region["anatomical"][index])))
        files[row["record_path"]] = row["record_sha256"]
        files[row["crops"]["518"]["path"]] = row["crops"]["518"]["sha256"]
        counts[row["dataset"]] += 1
    for filename, expected in files.items():
        if time.monotonic()-begun > 1800:
            raise TimeoutError("metadata input-hashing cap")
        if digest(Path(filename)) != expected:
            raise ValueError("regional study input binding mismatch")
    upstream_sha = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True, timeout=15).stdout.strip()
    upstream_dirty = subprocess.run(["git", "-C", str(upstream), "status", "--porcelain"], check=True,
                                    capture_output=True, text=True, timeout=15).stdout.strip()
    if upstream_sha != DINO_REVISION or upstream_dirty:
        raise ValueError("clean already-acquired official DINO source required")
    usable = {preparation.identity(row) for row in rows if row["detector_status"] == "success"}
    inventory = [{**{key: context[key] for key in ("target", "seed")}, "arm": arm,
                  "role_usable_counts": {role: sum(key in usable for key in context[role])
                                         for role in ("fitting", "validation", "calibration", "evaluation")}}
                 for context in partitions["contexts"] for arm in ARMS]
    plan = {"study": "regional_dino_v1", "scientific_execution_authorized": False,
            "fas_root": str(fas_root.resolve()), "source_revision": SOURCE_REVISION,
            "scope": SCOPE, "budget": BUDGET, "policy": POLICY, "head_settings": HEAD_SETTINGS,
            "arms": list(ARMS), "seeds": list(SEEDS), "dimensions": DIMENSIONS,
            "files": files, "weights": str(weights), "upstream": str(upstream),
            "upstream_revision": DINO_REVISION, "runtime": runtime(),
            "implementation_sha256": implementation(), "inventory": inventory,
            "usable_by_domain": dict(counts), "minimum_region_patch_counts": minima,
            "shared_geometry_failures": failures,
            "geometry_failure_counts": dict(Counter(failures.values())),
            "feature_forwards": len(usable)+2, "metadata_image_decodes": 0,
            "metadata_model_forwards": 0, "metadata_head_fits": 0}
    output.mkdir(parents=True)
    write_json(output/"plan.json", plan)
    return output/"plan.json"


def checked_plan(path: Path, sha: str) -> dict[str, Any]:
    require_private_output(path)
    plan = read_json(path, sha)
    expected = {"study": "regional_dino_v1", "source_revision": SOURCE_REVISION,
                "scientific_execution_authorized": False, "scope": SCOPE, "budget": BUDGET,
                "policy": POLICY, "head_settings": HEAD_SETTINGS, "arms": list(ARMS),
                "seeds": list(SEEDS), "dimensions": DIMENSIONS, "runtime": runtime(),
                "implementation_sha256": implementation(), "upstream_revision": DINO_REVISION}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise PermissionError("fixed complete regional DINO protocol, runtime and code required")
    return plan


def require_approval(path: Path, sha: str, approval_path: Path) -> dict[str, Any]:
    plan = checked_plan(path, sha)
    require_private_output(approval_path)
    approval = read_json(approval_path)
    expected = {"approved": True, "plan_sha256": sha, "scope": SCOPE, "budget": BUDGET,
                "no_automatic_retry": True}
    if any(approval.get(key) != value for key, value in expected.items()):
        raise PermissionError("explicit exact-plan owner approval required before scientific access")
    if (path.parent/"started.json").exists():
        raise FileExistsError("regional DINO attempt already started; no retry")
    return plan


def verify_inputs(plan: dict[str, Any], guard: Callable[[], None]) -> None:
    source_state(Path(plan["fas_root"]))
    for filename, expected in plan["files"].items():
        guard()
        if digest(Path(filename)) != expected:
            raise ValueError("sealed regional study input drift")
    upstream = Path(plan["upstream"])
    result = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True, timeout=15)
    dirty = subprocess.run(["git", "-C", str(upstream), "status", "--porcelain"],
                           check=True, capture_output=True, text=True, timeout=15)
    if result.stdout.strip() != DINO_REVISION or dirty.stdout.strip():
        raise ValueError("DINO upstream drift")


def extract(plan: dict[str, Any], original_rows: list[dict[str, Any]], services: dict[str, Any],
            root: Path, guard: Callable[[], None]) -> tuple[list[str], dict[str, np.ndarray]]:
    from .mat_preview import configure_mat_gpu
    write_json(root/"device.json", configure_mat_gpu(plan))
    sys.path.insert(0, plan["upstream"])
    factories = importlib.import_module("dinov2.hub.backbones")
    if not Path(factories.__file__).resolve().is_relative_to(Path(plan["upstream"])):
        raise ValueError("unbound regional DINO model import")
    torch.manual_seed(20261104)
    model = factories.dinov2_vitb14_reg(pretrained=False)
    model.load_state_dict(torch.load(plan["weights"], map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False).eval().cuda()
    before = tensor_digest(model.state_dict())
    frozen = services["fas.frozen_features"]

    def forward(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        guard()
        value = frozen.normalize_rgb(rgb, 518, (.485, .456, .406), (.229, .224, .225))
        with torch.inference_mode():
            output = model.forward_features(torch.from_numpy(value[None]).cuda())
        return (output["x_norm_clstoken"].cpu().numpy(),
                output["x_norm_patchtokens"].cpu().numpy())

    synthetic = np.random.default_rng(20261104).integers(0, 256, (518, 518, 3), dtype=np.uint8)
    first, second = forward(synthetic), forward(synthetic)
    frozen.parity(frozen.dino_pool(*first), frozen.dino_pool(*second))
    for left, right in zip(first, second, strict=True):
        if not np.array_equal(left, right):
            raise ValueError("deterministic regional DINO token replay failed")
    original_ids, original_features = services["paper321.runner"].load_features(original_rows, {}, guard)
    reference = dict(zip(original_ids, original_features["dino_reg"], strict=True))
    del original_features
    selected = sorted((row for row in original_rows if row["detector_status"] == "success"
                       and services["fas.source_fitting"].identity(row) not in plan["shared_geometry_failures"]),
                      key=services["fas.source_fitting"].identity)
    arrays = {name: np.empty((len(selected), DIMENSIONS[name]), dtype=np.float32)
              for name in REPRESENTATIONS}
    max_difference = 0.
    for index, row in enumerate(selected):
        guard()
        crop = row["crops"]["518"]
        payload = Path(crop["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != crop["sha256"]:
            raise ValueError("regional RGB518 crop changed before decode")
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (518, 518, 3) or bgr.dtype != np.uint8:
            raise ValueError("exact existing lossless RGB518 crop required; no resize fallback")
        cls, patches = forward(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pooled = pool_tokens(cls, patches, region_weights(crop_landmarks(transaction_for(row))))
        key = services["fas.source_fitting"].identity(row)
        max_difference = max(max_difference, frozen.parity(pooled["global"], reference[key][None]))
        for name in REPRESENTATIONS:
            arrays[name][index] = pooled[name][0]
        if index % 100 == 0:
            print({"phase": "regional_features", "completed": index+1, "total": len(selected)}, flush=True)
    if tensor_digest(model.state_dict()) != before:
        raise ValueError("frozen regional DINO weights changed")
    del model, reference
    gc.collect()
    torch.cuda.empty_cache()
    files = {}
    for name, value in arrays.items():
        path = root/f"{name}.npy"
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        files[path.name] = digest(path)
    ids = [services["fas.source_fitting"].identity(row) for row in selected]
    write_json(root/"features.json", {"ids": ids, "files_sha256": files,
                                    "frozen_state_sha256": before,
                                    "max_global_legacy_absolute_difference": max_difference,
                                    "exact_synthetic_token_replay": True})
    return ids, arrays


def paired_intervals(predictions: dict[str, np.ndarray], originals: list[dict[str, Any]],
                     policies: dict[str, Any], services: dict[str, Any],
                     guard: Callable[[], None]) -> list[dict[str, Any]]:
    from fas.fusion_study import components
    study = services["study"]
    identity = services["fas.source_fitting"].identity
    mapping = components([{**row, "group_id": row["subject_id"] or row["group_id"]} for row in originals])
    groups = sorted(set(mapping.values()))
    lookup = {group: index for index, group in enumerate(groups)}
    indexes = np.asarray([lookup[mapping[identity(row)]] for row in originals])
    rng = np.random.default_rng(POLICY["bootstrap_seed"])
    metrics = ("FA", "BFA", "AUROC_usable")
    samples = {(arm, metric): [] for arm in ARMS for metric in metrics}
    differences = {(comparison, left, right, metric): []
                   for comparison, left, right in CONTRASTS for metric in metrics}
    for _ in range(POLICY["bootstrap_repetitions"]):
        guard()
        weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)))[indexes]
        current = {arm: study.metrics(predictions[arm], originals, policies[arm]["1/20"]["threshold"], weights)
                   for arm in ARMS}
        for (arm, metric), values in samples.items():
            if current[arm][metric] is not None:
                values.append(current[arm][metric])
        for (_, arm, reference, metric), values in differences.items():
            left, right = current[arm][metric], current[reference][metric]
            if left is not None and right is not None:
                values.append(left-right)
    result = []
    records = [("point", arm, metric, values) for (arm, metric), values in samples.items()]
    records.extend((comparison, arm, metric, values)
                   for (comparison, arm, _, metric), values in differences.items())
    for comparison, arm, metric, values in records:
        result.append({"arm": arm, "metric": metric, "comparison": comparison,
                       "valid_repetitions": len(values), "declared_groups": len(groups),
                       "repetitions": POLICY["bootstrap_repetitions"],
                       "percentile95": np.quantile(values, [.025, .975]).tolist()
                       if len(values) >= .95*POLICY["bootstrap_repetitions"] else None})
    return result


def run_study(path: Path, sha: str, approval_path: Path) -> None:
    plan = require_approval(path, sha, approval_path)
    root = path.parent
    started, supervisor_pid = time.monotonic(), os.getppid()
    write_json(root/"started.json", {"plan_sha256": sha, "approval_sha256": digest(approval_path),
                                   "started_monotonic": started,
                                   "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
    deadline = started+BUDGET["max_seconds"]-30

    def guard() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("regional study shared wall-clock cap")
        if os.getppid() != supervisor_pid:
            raise RuntimeError("regional study supervisor disappeared; stopping orphan worker")

    verify_inputs(plan, guard)
    services, original_rows, partitions = inputs(Path(plan["fas_root"]))
    rows, failures = regional_population(original_rows)
    if failures != plan["shared_geometry_failures"]:
        raise ValueError("shared geometry-failure population drift")
    preparation = services["prepare_paper319"]
    expected_files = {str(preparation.MANIFEST): preparation.MANIFEST_SHA,
                      str(preparation.OUTPUT/"partitions.json"): PARTITIONS_SHA,
                      str(Path.home()/".cache/fas/source-pilot295-v1/checkpoints/dino_reg.pth"): WEIGHT_SHAS[0]}
    for row in original_rows:
        if row["detector_status"] == "success":
            expected_files[row["record_path"]] = row["record_sha256"]
            expected_files[row["crops"]["518"]["path"]] = row["crops"]["518"]["sha256"]
    if plan["files"] != expected_files or plan["weights"] not in expected_files:
        raise ValueError("complete original crop/transaction/checkpoint bindings required")
    usable = {services["fas.source_fitting"].identity(row) for row in rows
              if row["detector_status"] == "success"}
    expected_inventory = [{**{key: context[key] for key in ("target", "seed")}, "arm": arm,
                           "role_usable_counts": {role: sum(key in usable for key in context[role])
                               for role in ("fitting", "validation", "calibration", "evaluation")}}
                          for context in partitions["contexts"] for arm in ARMS]
    if plan["inventory"] != expected_inventory:
        raise ValueError("exact 72-head inventory required")
    ids, features = extract(plan, original_rows, services, root, guard)
    position = {key: index for index, key in enumerate(ids)}
    fitting, study = services["fas.source_fitting"], services["study"]
    indexed = {fitting.identity(row): row for row in rows}
    cases, intervals, files = [], [], {}
    for context in partitions["contexts"]:
        guard()
        pools = {role: [indexed[key] for key in context[role]]
                 for role in ("fitting", "validation", "calibration", "evaluation")}
        good = {role: [row for row in pool if row["detector_status"] == "success"]
                for role, pool in pools.items()}
        matrices = {role: np.asarray([position[fitting.identity(row)] for row in pool])
                    for role, pool in good.items()}
        labels = {role: np.asarray([row["binary_label"] == "attack" for row in pool], dtype=int)
                  for role, pool in good.items()}
        domains = {role: [row["dataset"] for row in pool] for role, pool in good.items()}
        models, prediction, policies = {}, {}, {}
        for arm in ARMS:
            guard()
            representation, objective = arm.split("__")
            x = features[representation]
            arguments = (x[matrices["fitting"]], labels["fitting"], domains["fitting"],
                         x[matrices["validation"]], labels["validation"], domains["validation"])
            options = {"seed": context["seed"], "settings": HEAD_SETTINGS, "guard": guard}
            head = (fitting.fit_head(*arguments, **options) if objective == "erm"
                    else fit_group_dro(*arguments, fitting=fitting, **options))
            calibration_logits = fitting.head_logits(head, x[matrices["calibration"]])
            calibration = fitting.fit_calibrator(calibration_logits, labels["calibration"], domains["calibration"], guard)
            calibration_scores = fitting.calibrate(calibration, calibration_logits)
            policies[arm] = {str(alpha): study.threshold(calibration_scores, good["calibration"], alpha)
                             for alpha in study.ALPHAS}
            prediction[arm] = fitting.calibrate(calibration, fitting.head_logits(head, x[matrices["evaluation"]]))
            models[arm] = {"head": head, "calibrator": calibration,
                           "calibration_scores": calibration_scores.tolist()}
            for alpha in study.ALPHAS:
                point = study.metrics(prediction[arm], pools["evaluation"], policies[arm][str(alpha)]["threshold"])
                point["FA_safe"] = point["accepted_attack"] <= int(point["original_attack"])*alpha.numerator//alpha.denominator
                point["joint_utility"] = point["FA_safe"] and point["BFA"] >= .5
                hindsight = services["oracle316"].oracle(
                    prediction[arm], labels["evaluation"], int(point["original_attack"]),
                    int(point["original_bona_fide"]), alpha=alpha)
                cases.append({"target": context["target"], "seed": context["seed"], "arm": arm,
                              "alpha": str(alpha), "policy": point, "oracle": hindsight})
            print({"phase": "regional_heads", "target": context["target"],
                   "seed": context["seed"], "arm": arm}, flush=True)
        name = f"context-{context['target']}-{context['seed']}.json"
        write_json(root/name, {"models": models, "predictions": {k: v.tolist() for k, v in prediction.items()},
                              "policies": policies, "evaluation_ids": context["evaluation"]})
        files[name] = digest(root/name)
        for record in paired_intervals(prediction, pools["evaluation"], policies, services, guard):
            intervals.append({"target": context["target"], "seed": context["seed"], **record})
    if len(cases) != 216:
        raise ValueError("complete six-arm/12-context/three-alpha output required")
    write_json(root/"result.json", {"plan_sha256": sha, "cases": cases, "conditional_intervals": intervals,
                                  "context_files_sha256": files, "feature_receipt_sha256": digest(root/"features.json"),
                                  "oracle_target_labels_used_for_selection": False})
    guard()
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2]/"scripts/regional_dino.py"),
                    "verify", "--plan", str(path), "--plan-sha256", sha],
                   check=True, timeout=max(1., deadline-time.monotonic()))


def summarize(path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    if not supervision.get("completed"):
        return {"completed": False, "supervision": supervision, "scientific_success_claimed": False}
    result = read_json(path.parent/"result.json")
    if result["plan_sha256"] != digest(path) or len(result["cases"]) != 216:
        raise ValueError("complete bound regional result required")
    for filename, sha in result["context_files_sha256"].items():
        if digest(path.parent/filename) != sha:
            raise ValueError("regional context evidence drift")
    if digest(path.parent/"features.json") != result["feature_receipt_sha256"]:
        raise ValueError("regional feature receipt drift")
    for filename, sha in read_json(path.parent/"features.json")["files_sha256"].items():
        if digest(path.parent/filename) != sha:
            raise ValueError("regional feature array drift")
    verification = read_json(path.parent/"verification.json")
    if (verification.get("completed") is not True
            or verification["result_sha256"] != digest(path.parent/"result.json")
            or verification["plan_sha256"] != digest(path)):
        raise ValueError("complete separate-process saved-feature reconstruction required")
    return {"completed": True, "plan_sha256": digest(path), "result": result,
            "supervision": supervision, "independent_reconstruction_completed": True,
            "verification": verification}
