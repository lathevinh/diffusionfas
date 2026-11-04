"""Separate CPU saved-feature reconstruction, not another scientific fit."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from .clip_dino import (
    ARMS, CONTRASTS, DIAGNOSTIC, DIMENSION, FRAMES, SEED, SETTINGS, SYSTEMS,
    balanced_weights, initial_head, logits, shuffled_features,
)
from .clip_study import BUDGET, ROLES, inputs, require_approval, source_pools, study_rows, verify_inputs
from .gpu_probe import tensor_digest
from .io import digest, read_json, write_json
from .regional_verify import independent_intervals

TOLERANCE = 1e-10


def independent_points(scores: np.ndarray, originals: list[dict[str, Any]], threshold: float,
                       alpha) -> dict[str, Any]:
    good = [row for row in originals if row["detector_status"] == "success"]
    y = np.asarray([row["binary_label"] == "attack" for row in good])
    n_attack = sum(row["binary_label"] == "attack" for row in originals)
    n_real = len(originals)-n_attack
    if scores.shape != (len(good),) or not np.isfinite(scores).all() or not y.any() or y.all():
        raise ValueError("finite exact usable score join and both target classes required")
    accepted = scores < threshold
    attack, real = int((accepted & y).sum()), int((accepted & ~y).sum())
    safe = attack <= n_attack*alpha.numerator//alpha.denominator
    return {"original_attack": n_attack, "original_bona_fide": n_real,
            "accepted_attack": attack, "accepted_bona_fide": real,
            "FA": attack/n_attack, "BFA": real/n_real, "AUROC_usable": roc_auc_score(y, scores),
            "usable_attack": int(y.sum()), "usable_bona_fide": int((~y).sum()),
            "FA_usable": attack/int(y.sum()), "BF_rejection_usable": 1-real/int((~y).sum()),
            "FA_within_constraint": safe, "utility_feasible": safe and real/n_real >= .5}


def independent_threshold(scores: np.ndarray, rows: list[dict[str, Any]], alpha) -> float:
    thresholds = []
    for domain in sorted({row["dataset"] for row in rows}):
        attacks = sorted(float(score) for score, row in zip(scores, rows, strict=True)
                         if row["dataset"] == domain and row["binary_label"] == "attack")
        if not attacks:
            raise ValueError("each calibration source needs successful attacks")
        thresholds.append(attacks[len(attacks)*alpha.numerator//alpha.denominator])
    return min(thresholds)


def require_close(left, right, message: str) -> None:
    if not np.allclose(left, right, rtol=0, atol=TOLERANCE):
        raise ValueError(message)


def verify_results(plan: dict[str, Any], root: Path, sha: str,
                   guard: Callable[[], None]) -> dict[str, Any]:
    services, rows, partitions = inputs(Path(plan["fas_root"]))
    if [c for c in partitions["contexts"] if c["seed"] == SEED] != plan["contexts"]:
        raise ValueError("verification context inventory changed")
    result = read_json(root/"result.json")
    if result["plan_sha256"] != sha:
        raise ValueError("result/plan binding mismatch")
    receipt = read_json(root/"features.json", result["features_sha256"])
    indexed = study_rows(rows, receipt)
    if (set(receipt["statuses"]) != set(indexed) or receipt["exact_synthetic_replay"] is not True
            or receipt["synthetic_encoder_frames"] != 8):
        raise ValueError("shared clip-status coverage and synthetic replay required")
    native_frames, encoded = 0, 0
    for index, entry in enumerate(plan["entries"]):
        guard()
        token, key = f"{index:05d}", entry["key"]
        success = receipt["statuses"][key]["success"]
        if entry["failure"] is not None:
            if success or receipt["statuses"][key]["failure"] != entry["failure"]:
                raise ValueError("historical/metadata failure was recovered or changed")
            continue
        decoded = read_json(root/"decode"/f"{token}.json", receipt["decode_sha256"][token])
        if (decoded["strict_success"] is not True or decoded["failure"] is not None
                or decoded["frame_count"] != entry["native_frame_count"]
                or decoded["frames_verified"] != entry["native_frame_count"]):
            raise ValueError("complete canonical full-decode identity replay required")
        native_frames += decoded["frames_verified"]
        if success:
            tracking = read_json(root/"tracking"/f"{token}.json", receipt["tracking_sha256"][token])
            if (tracking["key"] != key or np.asarray(tracking["boxes"]).shape != (FRAMES, 4)
                    or np.asarray(tracking["landmarks"]).shape != (FRAMES, 5, 2)
                    or len(tracking["geometry"]) != FRAMES):
                raise ValueError("successful clip tracking coverage mismatch")
    if native_frames != receipt["canonical_native_frames_verified"]:
        raise ValueError("native-frame receipt accounting mismatch")
    case_keys = {(c["target"], SEED, arm, str(alpha)) for c in plan["contexts"]
                 for arm in SYSTEMS for alpha in services["study"].ALPHAS}
    cases = {(c["target"], c["seed"], c["arm"], c["alpha"]): c for c in result["cases"]}
    if len(cases) != len(result["cases"]) or set(cases) != case_keys:
        raise ValueError("exact60 distinct operating cases required")
    interval_keys = {(c["target"], SEED, comparison, arm, metric) for c in plan["contexts"]
                     for comparison, arm, _ in ([("point", arm, None) for arm in SYSTEMS]+list(CONTRASTS))
                     for metric in ("FA", "BFA", "AUROC_usable")}
    intervals = {(c["target"], c["seed"], c["comparison"], c["arm"], c["metric"]): c
                 for c in result["conditional_intervals"]}
    if len(intervals) != len(result["conditional_intervals"]) or set(intervals) != interval_keys:
        raise ValueError("exact132 distinct interval records required")
    maximum_error = 0.
    for context in plan["contexts"]:
        guard()
        target = context["target"]
        if digest(root/f"{target}.npy") != receipt["files"][target]:
            raise ValueError("saved clip feature bytes changed")
        array = np.load(root/f"{target}.npy", mmap_mode="r", allow_pickle=False)
        expected_keys = sorted({key for role in ROLES for key in context[role]})
        positions = receipt["positions"][target]
        if positions != {key: i for i, key in enumerate(expected_keys)}:
            raise ValueError("fold-specific feature identity/order mismatch")
        if array.dtype != np.float32 or array.shape != (len(expected_keys), FRAMES, DIMENSION):
            raise ValueError("exact float32 N,16,768 feature contract required")
        for key, i in positions.items():
            good = indexed[key]["detector_status"] == "success"
            if not (np.isfinite(array[i]).all() if good else np.isnan(array[i]).all()):
                raise ValueError("no synthetic features for failed clips permitted")
            encoded += FRAMES*good
        saved = read_json(root/f"context-{target}.json", result["artifacts"][target])
        pools, keys, features = source_pools(context, indexed, positions, array)
        if saved["keys"] != keys or saved["target"] != target:
            raise ValueError("saved role/identity join changed")
        predictions, calibrations = {}, {}
        for arm in ARMS:
            guard()
            model_record = saved["models"][arm]
            head = root/"heads"/f"{target}-{arm}"/"head.pth"
            if digest(head) != model_record["head_sha256"]:
                raise ValueError("saved head changed")
            state = torch.load(head, map_location="cpu", weights_only=True)
            history = state["history"]
            values = [r["validation_BCE"] for r in history]
            if (state["arm"] != arm or state["seed"] != SEED or state["settings"] != SETTINGS
                    or len(history) != SETTINGS["epochs"]
                    or [r["epoch"] for r in history] != list(range(1, SETTINGS["epochs"]+1))
                    or not np.isfinite(values).all()
                    or state["selected_epoch"] != int(np.argmin(values))+1
                    or state["validation_BCE"] != min(values)
                    or model_record["selected_epoch"] != state["selected_epoch"]):
                raise ValueError("source-only fixed head training/selection inventory mismatch")
            model = initial_head(arm)
            if (tensor_digest(model.state_dict()) != tensor_digest(state["initial_state"])
                    or model_record["initial_state_sha256"] != tensor_digest(state["initial_state"])):
                raise ValueError("head initialization changed")
            model.load_state_dict(state["state_dict"], strict=True)
            model.eval()
            selected = ({role: shuffled_features(features[role], keys[role], target) for role in ROLES}
                        if arm == "shuffled" else features)
            labels = np.asarray([r["binary_label"] == "attack" for r in pools["validation"]], dtype=int)
            domains = [r["dataset"] for r in pools["validation"]]
            with torch.no_grad():
                value = balanced_weights(labels, domains) @ F.binary_cross_entropy_with_logits(
                    model(torch.from_numpy(selected["validation"].astype(np.float64))),
                    torch.from_numpy(labels.astype(np.float64)), reduction="none").numpy()
            require_close(value, state["validation_BCE"], "selected-state validation replay mismatch")
            calibrator = model_record["calibrator"]
            if calibrator["slope"] <= 0 or calibrator["optimizer_success"] is not True:
                raise ValueError("successful positive-affine source calibrator required")
            for role in ("calibration", "evaluation"):
                raw = logits(model, selected[role], guard)
                expected = np.asarray(model_record["probabilities"][role])
                rebuilt = expit(calibrator["slope"]*raw+calibrator["intercept"])
                if expected.shape != rebuilt.shape or not len(expected):
                    raise ValueError("exact reconstructed probability coverage required")
                error = float(np.max(np.abs(expected-rebuilt)))
                maximum_error = max(maximum_error, error)
                require_close(expected, rebuilt, "independent sigmoid probability reconstruction mismatch")
                native = services["fas.source_fitting"].calibrate(calibrator, raw)
                for policy in saved["policies"][arm].values():
                    if not np.array_equal(expected < policy["threshold"], native < policy["threshold"]):
                        raise ValueError("exact strict-threshold decisions changed")
                if role == "calibration":
                    calibrations[arm] = expected
                else:
                    predictions[arm] = expected
                    require_close(expected, saved["prediction"][arm], "saved prediction join changed")
            if arm == "temporal":
                raw = logits(model, shuffled_features(features["evaluation"], keys["evaluation"], target), guard)
                rebuilt = services["fas.source_fitting"].calibrate(calibrator, raw)
                require_close(rebuilt, saved["prediction"][DIAGNOSTIC], "inference-shuffle replay mismatch")
                predictions[DIAGNOSTIC] = np.asarray(saved["prediction"][DIAGNOSTIC])
                calibrations[DIAGNOSTIC] = calibrations[arm]
                if saved["policies"][DIAGNOSTIC] != saved["policies"][arm]:
                    raise ValueError("inference shuffle cannot refit calibration or policies")
        originals = [indexed[key] for key in context["evaluation"]]
        y = np.asarray([r["binary_label"] == "attack" for r in pools["evaluation"]], dtype=int)
        for arm in SYSTEMS:
            for alpha in services["study"].ALPHAS:
                threshold = independent_threshold(calibrations[arm], pools["calibration"], alpha)
                if saved["policies"][arm][str(alpha)]["threshold"] != threshold:
                    raise ValueError("independently sorted source threshold mismatch")
                case = cases[target, SEED, arm, str(alpha)]
                if case["diagnostic_only"] != (arm == DIAGNOSTIC):
                    raise ValueError("diagnostic order perturbation mislabeled")
                point = independent_points(predictions[arm], originals, threshold, alpha)
                for name, value in point.items():
                    require_close(case["policy"][name], value, "independent metric/utility mismatch")
                services["oracle316"].curve_check(case["oracle"], predictions[arm], y)
        group_map = {key: plan["groups"][key] for key in context["evaluation"]}
        rebuilt = independent_intervals(predictions, originals, saved["policies"], services,
                                         arms=SYSTEMS, contrasts=CONTRASTS, group_map=group_map)
        for record in rebuilt:
            expected = intervals[target, SEED, record["comparison"], record["arm"], record["metric"]]
            for field in ("valid_repetitions", "repetitions", "declared_groups"):
                if expected[field] != record[field]:
                    raise ValueError("independent interval support mismatch")
            if (expected["percentile95"] is None) != (record["percentile95"] is None):
                raise ValueError("independent interval validity mismatch")
            if record["percentile95"] is not None:
                require_close(expected["percentile95"], record["percentile95"], "independent bootstrap mismatch")
        del array, features
    if (encoded != receipt["production_encoder_frames"]
            or len(list((root/"heads").glob("*/started.json"))) != 16
            or len(list((root/"heads").glob("*/calibrator-started.json"))) != 16):
        raise ValueError("exact encoder/head/calibrator inventory mismatch")
    return {"completed": True, "plan_sha256": sha, "result_sha256": digest(root/"result.json"),
            "operating_cases": len(cases), "conditional_intervals": len(intervals),
            "maximum_probability_absolute_error": maximum_error, "tolerance": TOLERANCE,
            "new_fits": 0, "new_PAD_pixel_decodes": 0,
            "limitation": "separate_CPU_saved_feature_reconstruction_not_training_or_encoder_replication"}


def verify_completed(path: Path, sha: str, approval_path: Path) -> None:
    plan, approval = require_approval(path, sha, approval_path)
    root = path.parent
    start = read_json(root/"started.json")
    if start["worker_pid"] != os.getppid() or start["plan_sha256"] != sha:
        raise PermissionError("verification requires its original recorded worker parent")
    parent = os.getppid()
    deadline = approval["approved_monotonic"]+BUDGET["max_seconds"]-30
    write_json(root/"verification-started.json", {"plan_sha256": sha})

    def guard() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("verification shares the original clip approval clock")
        if os.getppid() != parent:
            raise RuntimeError("clip verifier lost its worker parent")

    verify_inputs(plan, guard)
    result = verify_results(plan, root, sha, guard)
    guard()
    write_json(root/"verification.json", result)
