"""Saved-score low-FA diagnosis and metadata-only square-crop feasibility."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
import time
from typing import Any

import numpy as np

from .celebahq_geometry import square_geometry
from .clip_verify import independent_points, independent_threshold, require_close
from .io import digest, read_json, require_private_output, write_json
from .preprocessing import MediaFailure
from .regional_study import inputs, transaction_for
from .spatial_dino import ARMS, SEED, SETTINGS
from .spatial_study import ROLES, population

PLAN_SHA = "41e5162ff9b44cc60de20509806291b228d4890751a3f96025d6599a6e9626db"
RESULT_SHA = "ba18be1b07c87684658723ee72ebe3f14fc3bc8b19cfa7ff424d6488b9c27f98"
AGGREGATE_SHA = "221f1c8a451d7293bd17d48897c2afd7a3576e8c7de65d2fb961928b7290dac3"
BUDGET = {"max_seconds": 1800, "max_rss_bytes": 2*1024**3, "max_output_bytes": 64*1024**2}
ALPHAS = (Fraction(1, 100), Fraction(1, 20))
CANDIDATES = ("expanded_legacy_square", "celebahq_style_square")
SUPPORT_MARGIN = 2.0


def grouped_oracle(scores: np.ndarray, labels: np.ndarray, original_attack: int,
                   original_real: int, alpha: Fraction) -> dict[str, Any]:
    """Enumerate strict-threshold score groups, not attack order statistics."""
    if (scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1)) or not np.isin(labels, (0, 1)).all()
            or type(original_attack) is not int or type(original_real) is not int
            or min(original_attack, original_real) <= 0
            or not isinstance(alpha, Fraction) or not 0 <= alpha < 1):
        raise ValueError("finite joined probabilities, binary labels and positive original counts required")
    attack_count = int(labels.sum())
    real_count = len(labels)-attack_count
    if attack_count > original_attack or real_count > original_real:
        raise ValueError("usable class counts exceed original denominators")
    values, inverse = np.unique(scores, return_inverse=True)
    attacks = np.bincount(inverse, weights=labels, minlength=len(values)).astype(np.int64)
    reals = np.bincount(inverse, weights=1-labels, minlength=len(values)).astype(np.int64)
    accepted_attack = np.r_[0, np.cumsum(attacks)]
    accepted_real = np.r_[0, np.cumsum(reals)]
    thresholds = np.r_[values, np.nextafter(1., 2.)]
    budget = original_attack*alpha.numerator//alpha.denominator
    choice = int(np.flatnonzero(accepted_attack <= budget)[-1])
    return {
        "attack_error_budget": budget, "accepted_attack": int(accepted_attack[choice]),
        "accepted_bona_fide": int(accepted_real[choice]),
        "FA": float(accepted_attack[choice]/original_attack),
        "max_BFA": float(accepted_real[choice]/original_real),
        "technical_failure_BFA_ceiling": real_count/original_real,
        "threshold_diagnostic_only": float(thresholds[choice]),
        "uses_target_labels": True, "deployable": False,
    }


def bottleneck(policy: dict, oracle: dict, beta: float) -> str:
    if oracle["technical_failure_BFA_ceiling"] < beta:
        return "technical_failure_ceiling"
    if oracle["max_BFA"] < beta:
        return "low_FA_separation_insufficient"
    if policy["FA_within_constraint"] and policy["BFA"] >= beta:
        return "current_policy_meets_this_empirical_goal"
    return "empirical_separation_available_policy_transfer_insufficient"


def check_operating_point(fresh: dict, saved: dict) -> None:
    for field in ("FA", "BFA", "accepted_attack", "accepted_bona_fide"):
        if fresh[field] != saved[field]:
            raise ValueError("saved operating point differs from independent recount")
    require_close(fresh["AUROC_usable"], saved["AUROC_usable"],
                  "saved AUROC differs from independent sklearn reconstruction")


def source_margin(quad: np.ndarray, width: int, height: int) -> float:
    if (quad.shape != (4, 2) or not np.isfinite(quad).all()
            or type(width) is not int or type(height) is not int or min(width, height) < 2):
        raise ValueError("four finite source corners and native dimensions required")
    return float(np.min(np.column_stack((quad[:, 0], quad[:, 1],
                                         width-1-quad[:, 0], height-1-quad[:, 1]))))


def crop_candidates(transaction: dict[str, Any]) -> dict[str, Any]:
    geometry, frame = transaction["geometry"], transaction["selected_frame"]
    bounds = np.asarray(geometry["crop_xyxy_half_open"], dtype=np.float64)
    matrix = np.asarray(geometry["alignment_matrix"], dtype=np.float64)
    landmarks = np.asarray(transaction["landmarks"], dtype=np.float64)
    if (bounds.shape != (4,) or matrix.shape != (2, 3) or landmarks.shape != (5, 2)
            or not all(np.isfinite(x).all() for x in (bounds, matrix, landmarks))
            or np.any(bounds != np.floor(bounds)) or np.any(bounds[2:] <= bounds[:2])
            or transaction["crop_shape"] != [int(bounds[3]-bounds[1]), int(bounds[2]-bounds[0]), 3]
            or not np.allclose(matrix[:, :2].T@matrix[:, :2], np.eye(2), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(matrix[:, :2]), 1., rtol=0, atol=1e-10)):
        raise ValueError("original finite rotation-only half-open crop geometry required")
    width, height = frame["width"], frame["height"]
    extent = bounds[2:]-bounds[:2]
    center = (bounds[:2]+bounds[2:]-1)/2
    signs = np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    inverse = np.linalg.inv(matrix[:, :2])
    old_quad = ((center+signs*(extent-1)/2)-matrix[:, 2])@inverse.T
    old_margin = source_margin(old_quad, width, height)
    side = float(max(extent))
    square_quad = ((center+signs*(side-1)/2)-matrix[:, 2])@inverse.T
    linear = matrix[:, :2]*(224/side)
    translation = (matrix[:, 2]-center)*(224/side)+111.5
    expanded = {"quad": square_quad, "matrix": np.column_stack((linear, translation)),
                "side": side, "render_matrix": np.column_stack(
                    (matrix[:, :2], matrix[:, 2]-center+(side-1)/2))}
    candidates = {}
    for name in CANDIDATES:
        if name == "expanded_legacy_square":
            candidate = expanded
        else:
            try:
                square = square_geometry(landmarks, 224)
            except MediaFailure as error:
                candidates[name] = {"status": "invalid_geometry", "reason": str(error),
                                    "new_crop_eligible": False, "strict_pair_eligible": False}
                continue
            candidate = {"quad": np.asarray(square["quad"]),
                         "matrix": np.asarray(square["source_to_output_matrix"]),
                         "side": square["side_source_pixels"]}
        margin = source_margin(candidate["quad"], width, height)
        transferred = landmarks@candidate["matrix"][:, :2].T+candidate["matrix"][:, 2]
        support = bool(np.all(transferred >= -.5) and np.all(transferred <= 223.5)
                       and np.linalg.norm(transferred[1]-transferred[0]) >= 1)
        eligible = margin >= SUPPORT_MARGIN and support
        candidates[name] = {
            "status": "eligible_conservative_metadata" if eligible else
                      "landmark_support_invalid" if not support else "real_pixel_margin_insufficient",
            "minimum_source_margin_pixels": margin, "side_source_pixels": candidate["side"],
            "source_to_RGB224_matrix": candidate["matrix"].tolist(),
            "source_quad": candidate["quad"].tolist(), "landmarks224": transferred.tolist(),
            "new_crop_eligible": eligible,
            "strict_pair_eligible": eligible and old_margin >= SUPPORT_MARGIN,
        }
        if name == "expanded_legacy_square":
            candidates[name]["source_to_square_render_matrix"] = candidate["render_matrix"].tolist()
    return {
        "native_width": width, "native_height": height,
        "legacy_height_over_width": float(extent[1]/extent[0]),
        "legacy_anisotropy_factor": float(max(extent)/min(extent)),
        "legacy_source_margin_pixels": old_margin,
        "legacy_real_pixel_support_conservative": old_margin >= SUPPORT_MARGIN,
        "candidates": candidates,
    }


def diagnosis(plan: dict, result: dict, saved_contexts: dict, indexed: dict,
              services: dict, guard: Callable) -> list[dict]:
    records = []
    cases = {(c["target"], c["arm"], c["alpha"]): c for c in result["cases"]}
    if len(cases) != 60 or len(result["cases"]) != 60:
        raise ValueError("exact completed sixty-case inventory required")
    for context in plan["contexts"]:
        target = context["target"]
        saved = saved_contexts[target]
        originals = [indexed[key] for key in context["evaluation"]]
        good_keys = [key for key in context["evaluation"] if indexed[key]["detector_status"] == "success"]
        if saved["keys"]["evaluation"] != good_keys or saved["target"] != target:
            raise ValueError("exact saved target score identities required")
        calibration = [indexed[key] for key in saved["keys"]["calibration"]]
        expected_calibration = [key for key in context["calibration"]
                                if indexed[key]["detector_status"] == "success"]
        if saved["keys"]["calibration"] != expected_calibration:
            raise ValueError("exact saved source calibration identities required")
        labels = np.asarray([indexed[key]["binary_label"] == "attack" for key in good_keys], dtype=int)
        for arm in ARMS:
            scores = np.asarray(saved["prediction"][arm], dtype=np.float64)
            if not np.array_equal(scores, saved["models"][arm]["probabilities"]["evaluation"]):
                raise ValueError("prediction/probability join differs")
            for alpha in ALPHAS:
                guard()
                threshold = independent_threshold(
                    np.asarray(saved["models"][arm]["probabilities"]["calibration"]), calibration, alpha)
                if threshold != saved["policies"][arm][str(alpha)]["threshold"]:
                    raise ValueError("saved threshold/source calibration reconstruction differs")
                policy = independent_points(scores, originals, threshold, alpha)
                previous = cases[target, arm, str(alpha)]
                check_operating_point(policy, previous["policy"])
                oracle = grouped_oracle(scores, labels, policy["original_attack"],
                                        policy["original_bona_fide"], alpha)
                for fresh, old in (("accepted_attack", "oracle_accepted_attack"),
                                   ("accepted_bona_fide", "oracle_accepted_bona_fide"),
                                   ("max_BFA", "oracle_max_BFA"), ("FA", "oracle_FA")):
                    if oracle[fresh] != previous["oracle"][old]:
                        raise ValueError("grouped oracle differs from saved order-statistic oracle")
                if policy["FA_within_constraint"] and policy["BFA"] > oracle["max_BFA"]:
                    raise ValueError("safe source policy exceeds empirical oracle")
                records.append({
                    "target": target, "arm": arm, "alpha": str(alpha), "source_policy": policy,
                    "oracle": oracle, "bottleneck_BFA50": bottleneck(policy, oracle, .5),
                    "bottleneck_BFA85_diagnostic_not_ISO": bottleneck(policy, oracle, .85),
                })
    if len(records) != 40:
        raise ValueError("exact forty low-FA diagnostic cases required")
    return records


def crop_inventory(plan: dict, indexed: dict, guard: Callable) -> tuple[list, dict]:
    records = []
    for entry in plan["entries"]:
        guard()
        row = indexed[entry["key"]]
        record = {"key": entry["key"], "dataset": row["dataset"], "label": row["binary_label"],
                  "historical_status": entry["status"], "geometry": None}
        if entry["status"] == "usable":
            record["geometry"] = crop_candidates(transaction_for(row))
        records.append(record)
    summaries = []
    for dataset in sorted({record["dataset"] for record in records}):
        for label in ("attack", "bona_fide"):
            selected = [r for r in records if (r["dataset"], r["label"]) == (dataset, label)]
            geometries = [r["geometry"] for r in selected if r["geometry"] is not None]
            ratios = [g["legacy_anisotropy_factor"] for g in geometries]
            summaries.append({
                "dataset": dataset, "label": label, "originals": len(selected),
                "current_usable": len(geometries),
                "legacy_anisotropy_median": float(np.median(ratios)),
                "legacy_anisotropy_p95": float(np.quantile(ratios, .95)),
                "legacy_real_pixel_support_conservative": sum(
                    g["legacy_real_pixel_support_conservative"] for g in geometries),
                "candidates": {name: {
                    "new_crop_eligible": sum(g["candidates"][name]["new_crop_eligible"] for g in geometries),
                    "strict_pair_eligible": sum(g["candidates"][name]["strict_pair_eligible"] for g in geometries),
                    "failures": dict(Counter(g["candidates"][name]["status"] for g in geometries
                                            if not g["candidates"][name]["new_crop_eligible"])),
                } for name in CANDIDATES},
            })
    by_key = {r["key"]: r for r in records}
    roles = []
    for context in plan["contexts"]:
        for name in CANDIDATES:
            counts = {}
            for role in ROLES:
                pool = [by_key[key] for key in context[role]]
                counts[role] = {label: {
                    "originals": sum(r["label"] == label for r in pool),
                    "strict_pair_eligible": sum(r["label"] == label and r["geometry"] is not None
                        and r["geometry"]["candidates"][name]["strict_pair_eligible"] for r in pool),
                } for label in ("attack", "bona_fide")}
            roles.append({"target": context["target"], "candidate": name, "roles": counts})
    return records, {"by_dataset_class": summaries, "by_fold_role_class": roles}


def completed_inputs(root: Path) -> tuple[dict, dict, dict, dict]:
    plan = read_json(root/"plan.json", PLAN_SHA)
    result = read_json(root/"result.json", RESULT_SHA)
    aggregate = read_json(root/"aggregate.json", AGGREGATE_SHA)
    completion, terminal = read_json(root/"completion.json"), read_json(root/"terminal.json")
    verification, supervision = read_json(root/"verification.json"), read_json(root/"supervision.json")
    if (not all(value["completed"] is True for value in
                (aggregate, completion, terminal, verification, supervision))
            or completion["aggregate_sha256"] != AGGREGATE_SHA
            or terminal["completion_sha256"] != digest(root/"completion.json")
            or completion["verification_sha256"] != digest(root/"verification.json")
            or completion["supervision_sha256"] != digest(root/"supervision.json")
            or verification["result_sha256"] != RESULT_SHA or verification["checkpoint_replays"] != 20
            or verification["operating_cases"] != 60 or verification["conditional_intervals"] != 132):
        raise ValueError("complete hash-bound spatial reconstruction required")
    saved = {context["target"]: read_json(root/f"context-{context['target']}.json",
                                         result["artifacts"][context["target"]])
             for context in plan["contexts"]}
    return plan, result, saved, {
        name: digest(root/name) for name in ("plan.json", "result.json", "aggregate.json",
                                            "verification.json", "supervision.json", "completion.json", "terminal.json")
    }


def run(root: Path, output: Path) -> dict:
    require_private_output(output/"summary.json")
    if output.exists():
        raise FileExistsError("new immutable diagnosis output root required")
    started = time.monotonic()

    def guard() -> None:
        if time.monotonic()-started > BUDGET["max_seconds"]-30:
            raise TimeoutError("spatial metadata diagnosis closure reserve")

    plan, result, saved, bindings = completed_inputs(root)
    services, rows, _ = inputs(Path(plan["fas_root"]))
    indexed = population(plan, rows, services)
    cases = diagnosis(plan, result, saved, indexed, services, guard)
    records, crops = crop_inventory(plan, indexed, guard)
    if len(records) != 7505 or sum(r["geometry"] is not None for r in records) != 7472:
        raise ValueError("exact original crop-feasibility inventory required")
    write_json(output/"cases.json", cases)
    write_json(output/"crop-records.json", records)
    result = {
        "completed": True, "scope": "saved_scores_and_native_geometry_metadata_only",
        "origin": bindings, "budget": BUDGET, "settings": SETTINGS,
        "diagnostic_cases": len(cases), "alphas": [str(a) for a in ALPHAS],
        "originals": len(records), "current_usable": 7472,
        "cases_sha256": digest(output/"cases.json"), "crop_records_sha256": digest(output/"crop-records.json"),
        "crop_summary": crops,
        "counts_BFA50": dict(Counter(case["bottleneck_BFA50"] for case in cases)),
        "crop_criterion": {
            "support_margin_source_pixels": SUPPORT_MARGIN,
            "conservative_metadata_only_actual_interpolation_support_pending": True,
            "no_padding_shifting_shrinking": True,
            "expanded_legacy_square": "same_rotation_center_max_dimension_expand_shorter_dimension",
            "celebahq_style_square": "existing_eye_mouth_geometry_diagnostic_alternative_not_DINO_training_requirement",
            "candidate_or_cohort_selection_authorized": False,
        },
        "PAD_pixel_decodes": 0, "new_training": 0, "calibration_fits": 0,
        "GPU_inference": 0, "patch_weight_maps": "not_computed_requires_bound_PAD_inference_scope",
        "oracle_deployable": False, "BFA85_is_internal_diagnostic_not_ISO": True,
        "wall_seconds": time.monotonic()-started,
    }
    guard()
    write_json(output/"summary.json", result)
    return result
