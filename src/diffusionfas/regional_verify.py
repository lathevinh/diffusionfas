"""CPU reconstruction of saved scores, operating counts and grouped intervals."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from .io import digest, read_json, write_json
from .regional_dino import ARMS, CONTRASTS, DIMENSIONS, HEAD_SETTINGS, REPRESENTATIONS
from .regional_study import BUDGET, POLICY, checked_plan, inputs, regional_population

TOLERANCE = 1e-10


def reconstruct_scores(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    coefficients = np.asarray(model["head"]["coefficients"], dtype=np.float64)
    calibration = model["calibrator"]
    if (features.ndim != 2 or coefficients.shape != (features.shape[1]+1,)
            or not np.isfinite(features).all() or not np.isfinite(coefficients).all()
            or not np.isfinite([calibration["slope"], calibration["intercept"]]).all()
            or calibration["slope"] <= 0):
        raise ValueError("finite matching saved head/calibrator and feature matrix required")
    logits = np.einsum("ij,j->i", features.astype(np.float64), coefficients[:-1], optimize=False)
    result = expit(calibration["slope"]*(logits+coefficients[-1])+calibration["intercept"])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite reconstructed regional probabilities")
    return result


def independent_intervals(predictions: dict[str, np.ndarray], originals: list[dict[str, Any]],
                          policies: dict[str, Any], services: dict[str, Any], *,
                          arms=ARMS, contrasts=CONTRASTS,
                          repetitions: int = POLICY["bootstrap_repetitions"],
                          group_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    from fas.fusion_study import components
    identity = services["fas.source_fitting"].identity
    mapping = (components([{**row, "group_id": row["subject_id"] or row["group_id"]} for row in originals])
               if group_map is None else group_map)
    if set(mapping) != {identity(row) for row in originals}:
        raise ValueError("exact original coverage required in bootstrap grouping")
    groups = sorted(set(mapping.values()))
    lookup = {group: index for index, group in enumerate(groups)}
    original_group = np.asarray([lookup[mapping[identity(row)]] for row in originals])
    original_y = np.asarray([row["binary_label"] == "attack" for row in originals])
    positions = np.asarray([i for i, row in enumerate(originals) if row["detector_status"] == "success"])
    y = original_y[positions]
    good_groups = original_group[positions]
    rng = np.random.default_rng(POLICY["bootstrap_seed"])
    draws = np.asarray([rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)))
                        for _ in range(repetitions)])
    attack_denominator = draws @ np.bincount(original_group, weights=original_y, minlength=len(groups))
    real_denominator = draws @ np.bincount(original_group, weights=~original_y, minlength=len(groups))
    values = {}
    for arm in arms:
        scores = predictions[arm]
        if scores.shape != (len(positions),) or not np.isfinite(scores).all():
            raise ValueError("complete finite paired saved scores required")
        accepted = scores < policies[arm]["1/20"]["threshold"]
        counts = {"FA": (accepted & y, attack_denominator),
                  "BFA": (accepted & ~y, real_denominator)}
        for metric, (mask, denominator) in counts.items():
            numerator = draws @ np.bincount(good_groups, weights=mask, minlength=len(groups))
            values[arm, metric] = np.divide(numerator, denominator,
                                            out=np.full(len(draws), np.nan), where=denominator > 0)
        order = np.argsort(scores, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(scores[order]))+1]
        weights = draws[:, good_groups[order]].astype(np.float64)
        attacks = np.add.reduceat(weights*y[order], starts, axis=1)
        real = np.add.reduceat(weights*~y[order], starts, axis=1)
        wins = (attacks*(np.cumsum(real, axis=1)-.5*real)).sum(axis=1)
        denominator = attacks.sum(axis=1)*real.sum(axis=1)
        values[arm, "AUROC_usable"] = np.divide(wins, denominator,
            out=np.full(len(draws), np.nan), where=denominator > 0)
    result = []
    comparisons: list[tuple[str, str, str | None]] = [("point", arm, None) for arm in arms]+list(contrasts)
    for comparison, arm, reference in comparisons:
        for metric in ("FA", "BFA", "AUROC_usable"):
            samples = values[arm, metric]
            if reference is not None:
                samples = samples-values[reference, metric]
            finite = samples[np.isfinite(samples)]
            result.append({"arm": arm, "metric": metric, "comparison": comparison,
                           "valid_repetitions": len(finite), "declared_groups": len(groups),
                           "repetitions": repetitions,
                           "percentile95": np.quantile(finite, [.025, .975]).tolist()
                           if len(finite) >= .95*repetitions else None})
    return result


def verify_completed(path: Path, sha: str) -> None:
    plan = checked_plan(path, sha)
    root = path.parent
    start = read_json(root/"started.json")
    if (start["plan_sha256"] != sha
            or start["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()):
        raise ValueError("original shared execution clock required for verification")
    deadline = start["started_monotonic"]+BUDGET["max_seconds"]-30
    parent = os.getppid()
    write_json(root/"verification-started.json", {"plan_sha256": sha})
    result = read_json(root/"result.json")
    if result["plan_sha256"] != sha:
        raise ValueError("verification result/plan binding mismatch")
    services, raw_rows, partitions = inputs(Path(plan["fas_root"]))
    rows, failures = regional_population(raw_rows)
    if failures != plan["shared_geometry_failures"]:
        raise ValueError("verification shared geometry failures changed")
    fitting = services["fas.source_fitting"]
    indexed = {fitting.identity(row): row for row in rows}
    receipt = read_json(root/"features.json", result["feature_receipt_sha256"])
    ids = sorted(key for key, row in indexed.items() if row["detector_status"] == "success")
    if receipt["ids"] != ids or receipt["exact_synthetic_token_replay"] is not True:
        raise ValueError("verification feature identity coverage mismatch")
    arrays = {}
    for representation in REPRESENTATIONS:
        filename = f"{representation}.npy"
        if digest(root/filename) != receipt["files_sha256"][filename]:
            raise ValueError("verification feature digest mismatch")
        array = np.load(root/filename, allow_pickle=False)
        if (array.shape != (len(ids), DIMENSIONS[representation]) or array.dtype != np.float32
                or not np.isfinite(array).all()):
            raise ValueError("verification feature contract mismatch")
        arrays[representation] = array
    for representation in REPRESENTATIONS[1:]:
        if not np.array_equal(arrays[representation][:, :1536], arrays["global"]):
            raise ValueError("regional/global feature prefixes differ")
    position = {key: i for i, key in enumerate(ids)}
    case_keys = {(context["target"], context["seed"], arm, str(alpha))
                 for context in partitions["contexts"] for arm in ARMS for alpha in services["study"].ALPHAS}
    cases = {(row["target"], row["seed"], row["arm"], row["alpha"]): row for row in result["cases"]}
    if len(cases) != len(result["cases"]) or set(cases) != case_keys:
        raise ValueError("verification requires exactly 216 distinct operating cases")
    comparisons: list[tuple[str, str, str | None]] = [("point", arm, None) for arm in ARMS]+list(CONTRASTS)
    interval_keys = {(context["target"], context["seed"], comparison, arm, metric)
                     for context in partitions["contexts"] for comparison, arm, _ in comparisons
                     for metric in ("FA", "BFA", "AUROC_usable")}
    intervals = {(row["target"], row["seed"], row["comparison"], row["arm"], row["metric"]): row
                 for row in result["conditional_intervals"]}
    if len(intervals) != len(result["conditional_intervals"]) or set(intervals) != interval_keys:
        raise ValueError("verification requires all 540 distinct primary interval records")
    if len(result["context_files_sha256"]) != 12:
        raise ValueError("verification requires twelve context records")
    maximum_error = 0.
    for context in partitions["contexts"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("verification exceeded original shared wall-clock budget")
        if os.getppid() != parent:
            raise RuntimeError("verification parent disappeared")
        filename = f"context-{context['target']}-{context['seed']}.json"
        saved = read_json(root/filename, result["context_files_sha256"][filename])
        if saved["evaluation_ids"] != context["evaluation"]:
            raise ValueError("verification target-original identity drift")
        originals = [indexed[key] for key in context["evaluation"]]
        good = {role: [indexed[key] for key in context[role] if key in position]
                for role in ("validation", "calibration", "evaluation")}
        matrices = {role: np.asarray([position[fitting.identity(row)] for row in pool])
                    for role, pool in good.items()}
        y = np.asarray([row["binary_label"] == "attack" for row in good["evaluation"]])
        original_attack = sum(row["binary_label"] == "attack" for row in originals)
        original_real = len(originals)-original_attack
        predictions = {}
        if any(set(saved[key]) != set(ARMS) for key in ("models", "predictions", "policies")):
            raise ValueError("verification requires all six saved arms")
        for arm in ARMS:
            representation, objective = arm.split("__")
            model = saved["models"][arm]
            head = model["head"]
            history = np.asarray(head["validation_history"])
            if (head["seed"] != context["seed"] or history.shape != (HEAD_SETTINGS["epochs"],)
                    or not np.isfinite(history).all()
                    or head["selected_epoch"] != int(history.argmin())+1
                    or head["validation_bce"] != history.min()):
                raise ValueError("verification source-only checkpoint/history mismatch")
            if objective == "group_dro":
                adversary = np.asarray(head["adversary_history"])
                if (adversary.shape != (200, 3) or not np.isfinite(adversary).all()
                        or np.any(adversary < 0) or not np.allclose(adversary.sum(axis=1), 1, atol=1e-12)):
                    raise ValueError("verification group-DRO history mismatch")
            x = arrays[representation]
            for role, scores in (("calibration", model["calibration_scores"]),
                                 ("evaluation", saved["predictions"][arm])):
                expected = np.asarray(scores)
                rebuilt = reconstruct_scores(x[matrices[role]], model)
                if expected.shape != rebuilt.shape or not np.isfinite(expected).all():
                    raise ValueError("verification probability coverage mismatch")
                error = float(np.max(np.abs(rebuilt-expected)))
                maximum_error = max(maximum_error, error)
                if error > TOLERANCE:
                    raise ValueError("verification independent probability reconstruction failed")
                if role == "evaluation":
                    predictions[arm] = expected
                    for policy in saved["policies"][arm].values():
                        if not np.array_equal(rebuilt < policy["threshold"], expected < policy["threshold"]):
                            raise ValueError("verification reconstructed decisions differ")
            for alpha in services["study"].ALPHAS:
                threshold = min(float(np.sort([
                    score for score, row in zip(model["calibration_scores"], good["calibration"], strict=True)
                    if row["dataset"] == domain and row["binary_label"] == "attack"])[
                        sum(row["dataset"] == domain and row["binary_label"] == "attack"
                            for row in good["calibration"])*alpha.numerator//alpha.denominator])
                    for domain in {row["dataset"] for row in good["calibration"]})
                if threshold != saved["policies"][arm][str(alpha)]["threshold"]:
                    raise ValueError("verification independent source threshold mismatch")
                case = cases[context["target"], context["seed"], arm, str(alpha)]
                accepted = predictions[arm] < threshold
                attacks, real = int((accepted & y).sum()), int((accepted & ~y).sum())
                metrics = {"original_attack": original_attack, "original_bona_fide": original_real,
                           "accepted_attack": attacks, "accepted_bona_fide": real,
                           "FA": attacks/original_attack, "BFA": real/original_real,
                           "usable_attack": int(y.sum()), "usable_bona_fide": int((~y).sum()),
                           "FA_usable": attacks/y.sum(), "BF_rejection_usable": 1-real/(~y).sum(),
                           "AUROC_usable": roc_auc_score(y, predictions[arm])}
                if any(not np.isclose(case["policy"][key], value, atol=TOLERANCE, rtol=0)
                       for key, value in metrics.items()):
                    raise ValueError("verification independent operating counts/AUROC mismatch")
                safe = attacks <= original_attack*alpha.numerator//alpha.denominator
                if case["policy"]["FA_safe"] != safe or case["policy"]["joint_utility"] != (safe and real/original_real >= .5):
                    raise ValueError("verification integer utility mismatch")
                services["oracle316"].curve_check(case["oracle"], predictions[arm], y.astype(int))
        for rebuilt in independent_intervals(predictions, originals, saved["policies"], services):
            key = context["target"], context["seed"], rebuilt["comparison"], rebuilt["arm"], rebuilt["metric"]
            expected = intervals[key]
            for name in ("valid_repetitions", "repetitions", "declared_groups"):
                if expected[name] != rebuilt[name]:
                    raise ValueError("verification bootstrap support mismatch")
            left, right = expected["percentile95"], rebuilt["percentile95"]
            if ((left is None) != (right is None)
                    or left is not None and not np.allclose(left, right, atol=TOLERANCE, rtol=0)):
                raise ValueError("verification independently vectorized bootstrap mismatch")
    write_json(root/"verification.json", {"completed": True, "plan_sha256": sha,
        "result_sha256": digest(root/"result.json"), "contexts": 12, "operating_cases": len(cases),
        "interval_records": len(intervals), "max_probability_absolute_error": maximum_error,
        "tolerance": TOLERANCE, "feature_ids": len(ids), "PAD_image_decodes": 0,
        "encoder_forwards": 0, "training_refits": 0,
        "scope": "separate_process_saved_feature_reconstruction_not_independent_training_replication"})
