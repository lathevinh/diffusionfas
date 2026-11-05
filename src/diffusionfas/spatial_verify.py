"""Separate checkpoint replay and independently reconstructed PAD statistics."""

from __future__ import annotations

from collections.abc import Callable
import gc
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit
import torch

from .clip_verify import independent_points, independent_threshold, require_close
from .gpu_probe import tensor_digest
from .io import digest, read_json, write_json
from .regional_study import inputs
from .regional_verify import independent_intervals
from .spatial_dino import (
    ARMS, SEED, SpatialReference, predict, source_modules, validate_state, validation_loss,
)
from .spatial_study import (
    BUDGET, CONTRASTS, EXECUTION_POLICY, ROLES, SCOPE, clock, loaders, population,
    require_approval, verify_inputs, verify_paired_traces,
)

DEVICE = "cuda"


def verify_results(plan: dict[str, Any], root: Path, sha: str, guard: Callable) -> dict[str, Any]:
    services, raw_rows, _ = inputs(Path(plan["fas_root"]))
    indexed = population(plan, raw_rows, services)
    source = source_modules(Path(plan["fas_root"]))
    load_rgb, load_prior = loaders(plan, guard)
    result = read_json(root/"result.json")
    if result["plan_sha256"] != sha:
        raise ValueError("spatial result/plan binding mismatch")
    case_keys = {(c["target"], SEED, arm, str(alpha)) for c in plan["contexts"]
                 for arm in ARMS for alpha in services["study"].ALPHAS}
    cases = {(c["target"], c["seed"], c["arm"], c["alpha"]): c for c in result["cases"]}
    if len(cases) != SCOPE["operating_cases"] or set(cases) != case_keys or len(cases) != len(result["cases"]):
        raise ValueError("exact sixty unique spatial operating cases required")
    interval_keys = {(c["target"], SEED, comparison, arm, metric) for c in plan["contexts"]
        for comparison, arm, _ in ([("point", arm, None) for arm in ARMS]+list(CONTRASTS))
        for metric in ("FA", "BFA", "AUROC_usable")}
    intervals = {(c["target"], c["seed"], c["comparison"], c["arm"], c["metric"]): c
                 for c in result["conditional_intervals"]}
    if (len(intervals) != SCOPE["conditional_intervals"] or set(intervals) != interval_keys
            or len(intervals) != len(result["conditional_intervals"])):
        raise ValueError("exact132 unique spatial interval records required")
    maximum_error, replays, expected_starts, expected_calibrators = 0., 0, set(), set()
    initial_reference, initial_fusion = None, None
    for context in plan["contexts"]:
        guard()
        target = context["target"]
        saved = read_json(root/f"context-{target}.json", result["artifacts"][target])
        pools = {role: [indexed[key] for key in context[role] if indexed[key]["detector_status"] == "success"]
                 for role in ROLES}
        keys = {role: [services["fas.source_fitting"].identity(row) for row in pool]
                for role, pool in pools.items()}
        if saved["keys"] != keys or saved["target"] != target or set(saved["models"]) != set(ARMS):
            raise ValueError("exact spatial role/key/model joins required")
        probabilities, step_counts, traces = {}, [], []
        for arm in ARMS:
            guard()
            folder = root/"models"/f"{target}-{arm}"
            expected_starts.add(folder/"started.json")
            expected_calibrators.add(folder/"calibrator-started.json")
            start = read_json(folder/"started.json")
            if start != {"target": target, "arm": arm, "seed": SEED, "plan_sha256": sha}:
                raise ValueError("exact training-start receipt required")
            if read_json(folder/"calibrator-started.json") != {"plan_sha256": sha}:
                raise ValueError("exact calibrator-start receipt required")
            record = saved["models"][arm]
            checkpoint, trace = folder/"model.pth", folder/"batches.jsonl"
            if digest(checkpoint) != record["checkpoint_sha256"] or digest(trace) != record["trace_sha256"]:
                raise ValueError("saved spatial checkpoint/trace bytes changed")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            validate_state(state, len(pools["fitting"]), arm)
            if (state["trace_sha256"] != digest(trace) or state["selected_epoch"] != record["selected_epoch"]
                    or state["optimizer_steps"] != record["optimizer_steps"]):
                raise ValueError("spatial saved-state/operating inventory mismatch")
            source.model.configure(SEED)
            model = SpatialReference(source.model.Reference(), arm).to(DEVICE)
            reference_sha = tensor_digest(model.reference.state_dict())
            fusion_sha = None if model.fusion is None else tensor_digest(model.fusion.state_dict())
            if (reference_sha != record["initial_reference_sha256"]
                    or fusion_sha != record["initial_fusion_sha256"]
                    or initial_reference not in (None, reference_sha)):
                raise ValueError("independently rebuilt original initialization differs")
            initial_reference = reference_sha
            if fusion_sha is not None:
                if initial_fusion not in (None, fusion_sha):
                    raise ValueError("initial fusion state differs across local arms")
                initial_fusion = fusion_sha
            model.load_state_dict(state["state_dict"], strict=True)
            logits = {role: predict(model, pools[role], load_rgb, load_prior, source, DEVICE, guard)
                      for role in ("validation", "calibration", "evaluation")}
            for role, values in logits.items():
                require_close(values, record["raw_logits"][role], "independent raw spatial logits differ")
            require_close(validation_loss(logits["validation"], pools["validation"]), state["validation_loss"],
                          "saved checkpoint fails its selected source-validation criterion")
            calibrator = record["calibrator"]
            if (not np.isfinite([calibrator["theta"], calibrator["intercept"], calibrator["slope"]]).all()
                    or calibrator["slope"] <= 0):
                raise ValueError("finite positive saved spatial calibrator required")
            require_close(calibrator["slope"], np.logaddexp(0, calibrator["theta"])+1e-6,
                          "saved calibration parameterization differs")
            rebuilt = {role: expit(calibrator["slope"]*logits[role]+calibrator["intercept"])
                       for role in ("calibration", "evaluation")}
            for role, values in rebuilt.items():
                stored = np.asarray(record["probabilities"][role])
                require_close(values, stored, "independent saved spatial probabilities differ")
                maximum_error = max(maximum_error, float(np.max(np.abs(values-stored))))
            require_close(rebuilt["evaluation"], saved["prediction"][arm], "spatial prediction join differs")
            # Use stored FP64 probabilities after numerical replay parity so exact
            # source order-statistic ties and the reported operating decisions agree.
            probabilities[arm] = np.asarray(saved["prediction"][arm], dtype=np.float64)
            calibration = np.asarray(record["probabilities"]["calibration"], dtype=np.float64)
            originals = [indexed[key] for key in context["evaluation"]]
            labels = np.asarray([row["binary_label"] == "attack" for row in pools["evaluation"]], dtype=int)
            for alpha in services["study"].ALPHAS:
                threshold = independent_threshold(calibration, pools["calibration"], alpha)
                if threshold != saved["policies"][arm][str(alpha)]["threshold"]:
                    raise ValueError("independently sorted spatial source threshold differs")
                replay_threshold = independent_threshold(rebuilt["calibration"], pools["calibration"], alpha)
                if not np.array_equal(rebuilt["evaluation"] < replay_threshold,
                                      probabilities[arm] < threshold):
                    raise ValueError("numerical replay changed a strict spatial operating decision")
                case = cases[target, SEED, arm, str(alpha)]
                point = independent_points(probabilities[arm], originals, threshold, alpha)
                for name, value in point.items():
                    require_close(case["policy"][name], value, f"independent spatial metric differs: {name}")
                services["oracle316"].curve_check(case["oracle"], probabilities[arm], labels)
                expected_oracle = services["oracle316"].oracle(probabilities[arm], labels,
                    point["original_attack"], point["original_bona_fide"], alpha=alpha)
                if case["oracle"] != expected_oracle:
                    raise ValueError("spatial oracle diagnostic metadata differs")
            step_counts.append(state["optimizer_steps"])
            traces.append(trace)
            replays += 1
            del model, state, logits
            gc.collect()
            torch.cuda.empty_cache()
            print("SPATIAL_CHECKPOINT_VERIFIED", target, arm, replays, flush=True)
        verify_paired_traces(traces, step_counts, guard)
        originals = [indexed[key] for key in context["evaluation"]]
        rebuilt = independent_intervals(probabilities, originals, saved["policies"], services,
            arms=ARMS, contrasts=CONTRASTS, repetitions=EXECUTION_POLICY["bootstrap_repetitions"],
            group_map={key: plan["groups"][key] for key in context["evaluation"]})
        for record in rebuilt:
            stored = intervals[target, SEED, record["comparison"], record["arm"], record["metric"]]
            for field in ("valid_repetitions", "repetitions", "declared_groups"):
                if record[field] != stored[field]:
                    raise ValueError("independent spatial interval support differs")
            if (record["percentile95"] is None) != (stored["percentile95"] is None):
                raise ValueError("independent spatial interval validity differs")
            if record["percentile95"] is not None:
                require_close(record["percentile95"], stored["percentile95"], "independent spatial bootstrap differs")
        guard()
    if (set((root/"models").glob("*/started.json")) != expected_starts
            or set((root/"models").glob("*/calibrator-started.json")) != expected_calibrators or replays != 20):
        raise ValueError("exact twenty training/calibrator starts and checkpoint replays required")
    return {"completed": True, "plan_sha256": sha, "result_sha256": digest(root/"result.json"),
            "checkpoint_replays": replays, "operating_cases": len(cases), "conditional_intervals": len(intervals),
            "maximum_probability_absolute_error": maximum_error, "tolerance": 1e-10,
            "new_fits": 0, "source_validation_recomputed": True, "PAD_crop_replay_required": True}


def verify_completed(path: Path, sha: str, approval_path: Path) -> None:
    plan, approval = require_approval(path, sha, approval_path)
    root = path.parent
    start = read_json(root/"started.json")
    if (start["worker_pid"] != os.getppid() or start["plan_sha256"] != sha
            or start["approval_sha256"] != digest(approval_path)):
        raise PermissionError("spatial reconstruction requires original worker parent and approval")
    parent = os.getppid()
    write_json(root/"verification-started.json", {"plan_sha256": sha, "verifier_pid": os.getpid()})

    def guard() -> None:
        clock(approval)
        if time.monotonic() >= approval["approved_monotonic"]+BUDGET["max_seconds"]-60:
            raise TimeoutError("spatial verification reached shared closure reserve")
        if os.getppid() != parent:
            raise RuntimeError("spatial verifier lost worker parent")

    verify_inputs(plan, guard)
    torch.cuda.set_per_process_memory_fraction(BUDGET["max_gpu_bytes"]/
                                               torch.cuda.get_device_properties(0).total_memory)
    result = verify_results(plan, root, sha, guard)
    guard()
    write_json(root/"verification.json", result)
