"""Opt-in source-validation attack diagnostic, never calibration or model selection."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import copy
from dataclasses import asdict, replace
import gc
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .adapter_pilot import ARMS as TRAINED_ARMS, configure_gpu
from .data import Video, addressed_seed, clip_start, load_manifest, window_indices
from .dataset import ClipCache
from .fas_import import MANIFEST_SHA
from .gpu_probe import tensor_digest
from .io import digest, object_digest, read_json, require_private_output, write_json
from .metrics import Prediction, auroc_summary, paired_auroc_bootstrap
from .models import build_model
from .native_replay import FasNativeReplay, decoder_policy
from .preprocessing import FasFaceProvider, MediaFailure, prepare_video
from .scoring import residual_profile
from .source_preparation import DETECTOR_SHA, item_video, selected_sources, source_media_entry
from .source_probe import SOURCES, stage_member
from .supervision import tree_bytes
from .training import check_source_pools, checkpoint_binding, install_trainable

ARMS = ("frozen", *TRAINED_ARMS)
AUTHORITY = {"source_validation_attacks": True, "model_selection": False,
             "threshold_fitting": False, "higher_score_is_attack": True}
BUDGET = {"max_seconds": 5400, "max_rss_bytes": 8 * 1024**3,
          "max_gpu_bytes": 8 * 1024**3, "max_output_bytes": 3 * 1024**3}


def validation_population(videos: list[Video]) -> list[Video]:
    return sorted((v for v in videos if v.dataset in SOURCES and v.partition == "validation"),
                  key=lambda v: (v.dataset, v.key))


def profile_map(payload: dict[str, Any], bonafide: list[Video], expected: dict[str, Any]) -> dict[str, list[float]]:
    rows = payload["records"]
    if len(rows) != len(bonafide) or {r["key"] for r in rows} != {v.key for v in bonafide}:
        raise ValueError("reused profiles must match every prepared validation bonafide original")
    domains = {v.key: v.dataset for v in bonafide}
    for row in rows:
        profile = np.asarray(row["profile"])
        if (row["dataset"] != domains[row["key"]] or profile.shape != (3,)
                or not np.isfinite(profile).all() or not math.isclose(
                    float(profile.mean()), row["loss"], rel_tol=1e-6, abs_tol=1e-8)):
            raise ValueError("invalid reused validation profile")
    grouped = {domain: [r["loss"] for r in rows if r["dataset"] == domain] for domain in SOURCES}
    measured = {domain: {"originals": len(values), "mean_loss": sum(values) / len(values)}
                for domain, values in grouped.items()}
    if (measured != expected["by_domain"]
            or sum(v["mean_loss"] for v in measured.values()) / 3 != expected["macro_loss"]):
        raise ValueError("reused validation profiles differ from published pilot aggregates")
    return {r["key"]: r["profile"] for r in rows}


def build_plan(manifest: Path, pilot_root: Path, source_manifest: Path, fas_root: Path,
               detector: Path, output: Path, *, source_sha: str = MANIFEST_SHA,
               detector_sha: str = DETECTOR_SHA) -> Path:
    require_private_output(output / "placeholder")
    if output.exists():
        raise ValueError("new source-diagnostic root required")
    pilot_plan = read_json(pilot_root / "plan.json")
    completion = read_json(pilot_root / "completion.json")
    supervision = read_json(pilot_root / "supervision.json")
    pilot_sha = digest(pilot_root / "plan.json")
    if (not completion["completed"] or not supervision["completed"]
            or completion["plan_sha256"] != pilot_sha or supervision["plan_sha256"] != pilot_sha
            or digest(manifest) != pilot_plan["manifest_sha256"]):
        raise ValueError("completed matching pilot and prepared population required")
    training = read_json(pilot_root / "training-result.json", completion["training_result_sha256"])
    replay = read_json(pilot_root / "replay.json", completion["replay_sha256"])
    if not replay["completed"] or replay["plan_sha256"] != pilot_sha:
        raise ValueError("completed pilot replay required")
    videos, provenance = load_manifest(manifest)
    if provenance["source_manifest_sha256"] != source_sha:
        raise ValueError("diagnostic source population provenance mismatch")
    fitting, validation, _ = check_source_pools(selected_sources(videos), "OULU-NPU")
    baseline = read_json(pilot_root / "baseline-private.json")
    profile_map(baseline, validation, training["baseline"])
    checkpoint_bindings = {}
    if tuple(a["variant"] for a in training["arms"]) != TRAINED_ARMS:
        raise ValueError("fixed spatial and temporal pilot arms required")
    for arm in training["arms"]:
        variant = arm["variant"]
        best_path, last_path = pilot_root / variant / "best.pt", pilot_root / variant / "last.pt"
        if digest(best_path) != arm["best_checkpoint_sha256"] or digest(last_path) != arm["last_checkpoint_sha256"]:
            raise ValueError("pilot checkpoint changed")
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        if (not last["complete"] or best["binding"] != checkpoint_binding(best["config"], fitting + validation)
                or best["binding"] != last["binding"] or best["best_epoch"] != last["best_epoch"]):
            raise ValueError("completed matching pilot checkpoint required")
        profile_map(best["validation"], validation, arm["best_validation"])
        checkpoint_bindings[variant] = {name: arm[f"{name}_checkpoint_sha256"] for name in ("best", "last")}
    source_rows = {f"{r['dataset']}/{r['video_id']}": r
                   for r in read_json(source_manifest, source_sha)["rows"]}
    entries = []
    for video in validation_population(videos):
        failure = None
        entry = {"video": video.to_dict()}
        if video.label == "bona_fide":
            if video.status != "prepared":
                failure = video.status.removeprefix("preparation_failure:")
        else:
            try:
                if video.status != "eligible_pending_tracking":
                    raise ValueError(video.status.removeprefix("preparation_failure:"))
                window_indices(video.timestamps, clip_start(video, 2))
                interval = video.interval
                if interval is None:
                    raise ValueError("insufficient_duration")
                duration = int((interval[1] - interval[0] + 1e-8) * 8) / 8
                entry["grid_frames"] = len(window_indices(video.timestamps, interval[0], duration, 8))
            except ValueError as error:
                failure = str(error)
            if failure is None:
                entry.update(source_media_entry(video, source_rows[video.key]))
        entry["metadata_failure"] = failure
        entries.append(entry)
    if not entries or not {e["video"]["label"] for e in entries} == {"bona_fide", "attack"}:
        raise ValueError("both source-validation classes required")
    config = copy.deepcopy(pilot_plan["config"])
    config["execution"] = {name: name in {"source_decode", "source_scoring"}
                           for name in config["execution"]}
    config["budget"] = {"max_seconds": BUDGET["max_seconds"], "max_output_bytes": BUDGET["max_output_bytes"],
                        "max_optimizer_steps": None}
    if digest(detector) != detector_sha:
        raise ValueError("diagnostic detector mismatch")
    estimate = (sum(e.get("grid_frames", 0) for e in entries) * 256 * 256 * 3
                + max(e.get("byte_size", 0) for e in entries) + 2 * manifest.stat().st_size)
    if estimate + 64 * 1024**2 > BUDGET["max_output_bytes"]:
        raise ValueError("diagnostic cache estimate exceeds approved new-output budget")
    project = Path(__file__).resolve().parents[2]
    plan = {
        "schema_version": 1, "scope": "source_validation_development_diagnostic",
        "authority": AUTHORITY, "budget": BUDGET, "config": config, "entries": entries,
        "manifest": str(manifest.resolve()), "manifest_sha256": digest(manifest),
        "pilot_root": str(pilot_root.resolve()), "pilot_plan_sha256": pilot_sha,
        "pilot_completion_sha256": digest(pilot_root / "completion.json"),
        "pilot_training_sha256": completion["training_result_sha256"],
        "pilot_baseline_sha256": digest(pilot_root / "baseline-private.json"),
        "checkpoints": checkpoint_bindings,
        "source_manifest": str(source_manifest.resolve()), "source_manifest_sha256": source_sha,
        "fas_root": str(fas_root.resolve()), "detector": str(detector.resolve()),
        "detector_sha256": detector_sha, "preprocessing_sha256": digest(fas_root / "src/fas/preprocessing.py"),
        "msu_reader_sha256": digest(fas_root / "scripts/inventory_msu.py"),
        "conservative_estimate_bytes": estimate,
        "implementation_sha256": {str(p.relative_to(project)): digest(p)
                                  for directory in ("src", "scripts")
                                  for p in sorted((project / directory).rglob("*.py"))},
    }
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def checked_inputs(path: Path, sha: str) -> tuple[dict[str, Any], list[Video]]:
    plan = read_json(path, sha)
    require_private_output(path)
    if plan["authority"] != AUTHORITY or plan["budget"] != BUDGET:
        raise PermissionError("fixed diagnostic-only authority and approved caps required")
    config = plan["config"]
    if (config["target"] != "OULU-NPU"
            or {k for k, v in config["execution"].items() if v} != {"source_decode", "source_scoring"}
            or config["scoring"] != {"alpha_bars": [0.9, 0.5, 0.1], "draws": 2}):
        raise PermissionError("diagnostic cannot train, calibrate, change residual rule or access target")
    project = Path(__file__).resolve().parents[2]
    for name, expected in plan["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("diagnostic implementation changed after plan")
    if digest(Path(plan["manifest"])) != plan["manifest_sha256"]:
        raise ValueError("diagnostic population changed")
    videos, _ = load_manifest(Path(plan["manifest"]))
    expected = [v.to_dict() for v in validation_population(videos)]
    if object_digest(expected) != object_digest([e["video"] for e in plan["entries"]]):
        raise ValueError("diagnostic must preserve entire declared validation population")
    read_json(Path(plan["source_manifest"]), plan["source_manifest_sha256"])
    pilot = Path(plan["pilot_root"])
    pilot_plan = read_json(pilot / "plan.json", plan["pilot_plan_sha256"])
    if (config["seed"] != pilot_plan["config"]["seed"]
            or config["clip"] != pilot_plan["config"]["clip"]
            or config["preprocessing"] != pilot_plan["config"]["preprocessing"]):
        raise ValueError("diagnostic input/noise/decoder contracts differ from locked pilot")
    read_json(pilot / "completion.json", plan["pilot_completion_sha256"])
    read_json(pilot / "training-result.json", plan["pilot_training_sha256"])
    read_json(pilot / "baseline-private.json", plan["pilot_baseline_sha256"])
    for variant, checkpoints in plan["checkpoints"].items():
        for name, expected_sha in checkpoints.items():
            if digest(pilot / variant / f"{name}.pt") != expected_sha:
                raise ValueError("locked diagnostic checkpoint changed")
    return plan, videos


def run_diagnostic(path: Path, sha: str) -> None:
    import cv2

    plan, original = checked_inputs(path, sha)
    root = path.parent
    if (root / "started.json").exists():
        raise ValueError("diagnostic already attempted; no automatic retry")
    write_json(root / "started.json", {"plan_sha256": sha})
    started = time.monotonic()
    deadline = started + BUDGET["max_seconds"] - 20
    config, fas_root = plan["config"], Path(plan["fas_root"])
    if digest(fas_root / "scripts/inventory_msu.py") != plan["msu_reader_sha256"]:
        raise ValueError("exact archive reader changed")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    provider = FasFaceProvider(fas_root, plan["preprocessing_sha256"], Path(plan["detector"]),
                               plan["detector_sha256"])
    decoders, helper_sha = decoder_policy(config)
    native = None
    if any(e["video"]["dataset"] == "MSU-MFSD" and e["video"]["label"] == "attack"
           and not e["metadata_failure"] for e in plan["entries"]):
        if helper_sha is None:
            raise ValueError("explicit strict MSU decoder helper required")
        native = FasNativeReplay(fas_root, helper_sha, Path(plan["source_manifest"]),
                                 plan["source_manifest_sha256"])
    records, caches, staging = root / "records", root / "caches", root / "staging"
    for directory in (records, caches, staging):
        directory.mkdir()
    prepared = []
    with ExitStack() as stack:
        msu_context: dict[str, Any] = {}
        for index, entry in enumerate(plan["entries"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic shared wall cap")
            video, failure = item_video(entry), entry["metadata_failure"]
            result = replace(video, status=f"preparation_failure:{failure}") if failure else video
            attempted = False
            if video.label == "attack" and not failure:
                attempted = True
                write_json(records / f"started-{index:06d}.json", {"index": index})
                staged = staging / f"source-{index:06d}{Path(entry['member']).suffix}"
                try:
                    stage_member(entry, fas_root, fas_root, staged, stack, msu_context, deadline,
                                 allow_validation_attacks=True)
                    replay = None
                    if decoders[video.dataset] == "fas_strict_ffmpeg":
                        if native is None:
                            raise ValueError("native replay unavailable")
                        replay = native.bind(replace(video, media_path=str(staged)),
                                             root / "native-replay" / f"original-{index:06d}",
                                             expected_receipt_sha256=entry["receipt_sha256"])
                    try:
                        result = prepare_video(
                            replace(video, media_path=str(staged)), provider, caches / f"{index:06d}.npz",
                            fps=8, max_seconds=min(180, max(0.001, deadline - time.monotonic())),
                            max_output_bytes=min(512 * 1024**2, BUDGET["max_output_bytes"] - tree_bytes(root)),
                            native_replay=replay,
                        )
                    except MediaFailure as error:
                        failure = str(error)
                        result = replace(video, status=f"preparation_failure:{failure}")
                finally:
                    staged.unlink(missing_ok=True)
            result = replace(result, media_path=None)
            prepared.append(result)
            write_json(records / f"result-{index:06d}.json", {
                "video": result.to_dict(), "failure": failure, "new_attack_preparation_attempted": attempted,
            })
            if tree_bytes(root) > BUDGET["max_output_bytes"]:
                raise OSError("diagnostic new-output cap")
            print({"phase": "preparation", "originals": index + 1, "total": len(plan["entries"]),
                   "status": result.status}, flush=True)
    write_json(root / "diagnostic-manifest.json", {
        "schema_version": 1, "videos": [v.to_dict() for v in prepared],
        "provenance": {"input_manifest_sha256": plan["manifest_sha256"], "plan_sha256": sha,
                       "development_diagnostic_only": True},
    })
    device = configure_gpu(plan)
    cache = ClipCache(256)
    pilot = Path(plan["pilot_root"])
    training = read_json(pilot / "training-result.json", plan["pilot_training_sha256"])
    bona = [v for v in prepared if v.label == "bona_fide" and v.status == "prepared"]
    representative_keys = {next(v.key for v in bona if v.dataset == domain) for domain in SOURCES}
    all_predictions = {}
    arm_reports = []
    for variant in ARMS:
        if variant == "frozen":
            model_config = copy.deepcopy(config)
            model_config["model"]["variant"] = "pretrained_spatial"
            payload = read_json(pilot / "baseline-private.json", plan["pilot_baseline_sha256"])
            reference = training["baseline"]
            checkpoint_sha = config["model"]["bundle_manifest_sha256"]
        else:
            best_path = pilot / variant / "best.pt"
            state = torch.load(best_path, map_location="cpu", weights_only=True)
            model_config, payload = state["config"], state["validation"]
            reference = next(a["best_validation"] for a in training["arms"] if a["variant"] == variant)
            checkpoint_sha = plan["checkpoints"][variant]["best"]
        reused_profiles = profile_map(payload, bona, reference)
        model = build_model(model_config).cuda().eval()
        model.backbone_frame_batch = 4
        if variant != "frozen":
            install_trainable(model, state["ema"])
        torch.cuda.reset_peak_memory_stats()
        predictions, private_rows = [], []
        replay_error = 0.0
        for index, video in enumerate(prepared):
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic shared wall cap")
            profile = None
            if video.status != "prepared":
                failure = video.status.removeprefix("preparation_failure:")
                score = None
            else:
                failure = None
                if video.label == "bona_fide":
                    profile = reused_profiles[video.key]
                if video.label == "attack" or video.key in representative_keys:
                    measured = residual_profile(
                        model, cache.get(video)[None].cuda(), [video.key], seed=config["seed"],
                        **config["scoring"],
                    )[0].cpu()
                    if profile is not None:
                        expected = torch.tensor(profile)
                        torch.testing.assert_close(measured, expected, rtol=1e-5, atol=1e-6)
                        replay_error = max(replay_error, float((measured - expected).abs().max()))
                    else:
                        profile = measured.tolist()
                if profile is None:
                    raise ValueError("prepared original has no observed residual profile")
                score = float(np.asarray(profile, dtype=np.float64).mean())
            prediction = Prediction(video.key, video.dataset, video.group, video.partition,
                                    video.label, score, failure, video.attack_family)
            predictions.append(prediction)
            private_rows.append({**asdict(prediction), "profile": profile})
        baseline_backbone = training["arms"][0]["frozen_backbone_sha256"]
        if tensor_digest(model.backbone.state_dict()) != baseline_backbone:
            raise ValueError("diagnostic frozen backbone differs from locked pilot")
        write_json(root / f"scores-{variant}.json", {
            "plan_sha256": sha, "checkpoint_sha256": checkpoint_sha,
            "higher_score_is_attack": True, "records": private_rows,
        })
        all_predictions[variant] = predictions
        arm_reports.append({"variant": variant, "checkpoint_sha256": checkpoint_sha,
                            "reused_bonafide_profiles": len(bona), "bonafide_replay_originals": 3,
                            "bonafide_replay_max_absolute_error": replay_error,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        print({"phase": "scoring", "variant": variant, "originals": len(predictions)}, flush=True)
        del model
        if variant != "frozen":
            del state
        gc.collect()
        torch.cuda.empty_cache()
    domains = {}
    for domain in SOURCES:
        arms = {name: [r for r in rows if r.dataset == domain] for name, rows in all_predictions.items()}
        reference_rows = next(iter(arms.values()))
        domains[domain] = {
            "arms": {name: auroc_summary(rows) for name, rows in arms.items()},
            "paired_group_bootstrap": paired_auroc_bootstrap(
                arms, repetitions=2000, seed=addressed_seed(20261103, domain, "diagnostic-bootstrap")),
            "grouping_unit": "declared_subject_group" if domain != "SiW-Mv2" else "original_group_subject_unknown",
            "failures_by_reason": dict(Counter(r.failure for r in reference_rows if r.failure)),
        }
    macro = {}
    for variant in ARMS:
        values = [domains[d]["arms"][variant]["auroc_usable"] for d in SOURCES]
        macro[variant] = (sum(value for value in values if value is not None) / 3
                          if all(value is not None for value in values) else None)
    write_json(root / "result.json", {
        "plan_sha256": sha, "domains": domains, "domain_macro_auroc_usable": macro,
        "macro_unavailable_reason": None if all(v is not None for v in macro.values()) else "missing_domain_auc",
        "arms": arm_reports, "device": device,
        "unknown_attack_family_originals": sum(v.label == "attack" and not v.attack_family for v in prepared),
        "bonafide_validation_previously_used_for_checkpoint_selection": True,
    })
    write_json(root / "completion.json", {"completed": True, "plan_sha256": sha,
                                        "result_sha256": digest(root / "result.json"),
                                        "manifest_sha256": digest(root / "diagnostic-manifest.json"),
                                        "wall_seconds": time.monotonic() - started})


def summarize(path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    plan, root = read_json(path), path.parent
    counts: dict[str, Counter] = {}
    for index, entry in enumerate(plan["entries"]):
        group = f"{entry['video']['dataset']}:{entry['video']['label']}"
        counts.setdefault(group, Counter())["originals"] += 1
        receipt = root / "records" / f"result-{index:06d}.json"
        if receipt.exists():
            record = read_json(receipt)
            if record["video"]["key"] != entry["video"]["key"]:
                raise ValueError("diagnostic preparation receipt identity mismatch")
            counts[group]["prepared" if record["video"]["status"] == "prepared" else "technical_failure"] += 1
        elif (root / "records" / f"started-{index:06d}.json").exists():
            counts[group]["interrupted"] += 1
        else:
            counts[group]["unattempted"] += 1
        staged = root / "staging" / f"source-{index:06d}{Path(entry.get('member', 'unused')).suffix}"
        staged.unlink(missing_ok=True)
    result, completion = None, None
    if (root / "completion.json").exists():
        completion = read_json(root / "completion.json")
        if (completion["plan_sha256"] != digest(path)
                or digest(root / "diagnostic-manifest.json") != completion["manifest_sha256"]):
            raise ValueError("diagnostic completion binding mismatch")
        result = read_json(root / "result.json", completion["result_sha256"])
        if result["plan_sha256"] != digest(path):
            raise ValueError("diagnostic result binding mismatch")
    return {
        "schema_version": 1, "scope": plan["scope"],
        "completed": bool(supervision["completed"] and completion is not None and completion["completed"]),
        "selection_plan_sha256": digest(path), "input_manifest_sha256": plan["manifest_sha256"],
        "pilot_plan_sha256": plan["pilot_plan_sha256"], "locked_checkpoints": plan["checkpoints"],
        "implementation_sha256": plan["implementation_sha256"],
        "budget": plan["budget"], "supervision": supervision, "new_output_bytes": tree_bytes(root),
        "original_denominator": len(plan["entries"]), "counts_by_domain_label": counts, "result": result,
        "score_rule": "float64_mean_of_three_fixed_residual_levels_higher_is_attack",
        "training_runs": 0, "calibration_video_decodes": 0, "calibration_model_forwards": 0,
        "target_video_decodes": 0, "target_model_forwards": 0,
        "threshold_fitted": False, "score_direction_changed": False, "model_selection_performed": False,
        "unbiased_test_result": False, "automatic_retries": 0, "selection_replacements": 0,
        "default_validation_attack_guards_unchanged": True, "native_RGB_waivers": 0,
    }
