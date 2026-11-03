"""Approved matched MSU crop diagnostic with one frozen DDPM, never training."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import copy
from dataclasses import asdict, replace
import importlib
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

from .adapter_pilot import configure_gpu
from .celebahq_geometry import REAL_PIXEL_POLICY, crop_real_pixels
from .crop_preview import checked_inputs as checked_preview, cache_samples, read_native_frames, real_pixel_selection
from .data import Video, addressed_seed, load_manifest
from .fas_import import MANIFEST_SHA
from .gpu_probe import tensor_digest
from .io import digest, read_json, require_private_output, write_json
from .metrics import Prediction, auroc_summary, paired_auroc_bootstrap
from .models import build_model
from .native_replay import FasNativeReplay, decoder_policy
from .preprocessing import MediaFailure
from .scoring import residual_profile
from .source_preparation import item_video, source_media_entry
from .source_probe import stage_member
from .supervision import tree_bytes

BUDGET = {"max_seconds": 1800, "max_rss_bytes": 8 * 1024**3,
          "max_gpu_bytes": 8 * 1024**3, "max_output_bytes": 1024**3}
AUTHORITY = {"source_validation_MSU_only": True, "frozen_model_only": True,
             "higher_score_is_attack": True, "training": False, "threshold_fitting": False,
             "fitting_pixels": False, "calibration_pixels": False, "target_pixels": False,
             "automatic_retry": False}
PREVIEW_SHA = "bc4014dd7caeb8baa49420c27ae6091f8e562a4417ad3333646c8f6248bc8e41"
SCORING = {"alpha_bars": [0.9, 0.5, 0.1], "draws": 2}
SEED = 20261103
ARMS = ("legacy", "real_pixels")


def full_middle_samples(video: Video) -> list[dict[str, Any]]:
    anchors = cache_samples(video)
    start = anchors[0]["cache_index"]
    with np.load(video.cache_path, allow_pickle=False) as cache:
        return [{"clip_slot": slot, "cache_index": start + slot,
                 "native_order": int(cache["native_orders"][start + slot]),
                 "native_timestamp": float(cache["native_timestamps"][start + slot]),
                 "box": cache["boxes"][start + slot].tolist(),
                 "landmarks": cache["landmarks"][start + slot].tolist()} for slot in range(16)]


def approved_preview(root: Path) -> tuple[dict[str, Any], list[Video]]:
    preview = checked_preview(root / "plan.json", PREVIEW_SHA)
    approval = read_json(root / "visual-approval.json")
    completion = read_json(root / "completion.json")
    supervision = read_json(root / "supervision.json")
    if (preview["policy"] != REAL_PIXEL_POLICY or approval["crop_accepted"] is not True
            or approval["preview_plan_sha256"] != PREVIEW_SHA or approval["budget"] != BUDGET
            or approval["next_run_scope"] != "22_MSU_validation_originals_frozen_DDPM_matched_crop_AB"
            or approval["html_sha256"] != digest(root / "review.html")
            or completion["html_sha256"] != approval["html_sha256"]
            or completion["plan_sha256"] != PREVIEW_SHA or supervision["plan_sha256"] != PREVIEW_SHA
            or not completion["completed"] or not supervision["completed"]
            or completion["counts"] != {"rendered": 6, "frames": 18}):
        raise PermissionError("completed strict preview, visual acceptance and approved budget required")
    videos, _ = load_manifest(Path(preview["manifest"]))
    _, cohort = real_pixel_selection(videos, Path(preview["real_pixel_cohort"]["audit_root"]))
    if cohort != preview["real_pixel_cohort"]:
        raise ValueError("approved MSU population changed")
    eligible = set(cohort["eligible_keys"])
    selected = sorted((v for v in videos if v.key in eligible), key=lambda v: v.key)
    validate_population(selected)
    return preview, selected


def validate_population(videos: list[Video]) -> None:
    if (len(videos) != 22 or len({v.key for v in videos}) != 22
            or Counter(v.label for v in videos) != {"bona_fide": 6, "attack": 16}
            or any(v.dataset != "MSU-MFSD" or v.partition != "validation" or v.status != "prepared"
                   for v in videos)):
        raise PermissionError("exact approved 22-original MSU validation cohort required")


def build_plan(preview_root: Path, output: Path) -> Path:
    require_private_output(output / "plan.json")
    if output.exists():
        raise ValueError("new crop-ablation root required")
    preview, selected = approved_preview(preview_root)
    diagnostic = read_json(Path(preview["diagnostic_root"]) / "plan.json", preview["diagnostic_plan_sha256"])
    config = copy.deepcopy(diagnostic["config"])
    if config["seed"] != SEED or config["scoring"] != SCORING:
        raise ValueError("existing seed and noise recipe must remain unchanged")
    config["model"]["variant"] = "pretrained_spatial"
    config["execution"] = {name: name in {"source_decode", "source_scoring"} for name in config["execution"]}
    config["budget"] = {"max_seconds": BUDGET["max_seconds"], "max_output_bytes": BUDGET["max_output_bytes"],
                        "max_optimizer_steps": None}
    rows = {f"{r['dataset']}/{r['video_id']}": r
            for r in read_json(Path(preview["source_manifest"]), MANIFEST_SHA)["rows"]}
    entries = [{"video": v.to_dict(), **source_media_entry(v, rows[v.key]),
                "samples": full_middle_samples(v)} for v in selected]
    estimate = len(entries) * (2 * 16 * 256 * 256 * 3 + 2 * 1024**2) + max(e["byte_size"] for e in entries)
    if estimate + 64 * 1024**2 > BUDGET["max_output_bytes"]:
        raise ValueError("conservative crop-pair output estimate exceeds approval")
    project = Path(__file__).resolve().parents[2]
    plan = {"scope": "MSU_frozen_matched_crop_development_diagnostic", "budget": BUDGET,
            "authority": AUTHORITY, "policy": REAL_PIXEL_POLICY, "config": config,
            "preview_root": str(preview_root.resolve()), "preview_plan_sha256": PREVIEW_SHA,
            "approval_sha256": digest(preview_root / "visual-approval.json"),
            "entries": entries, "cohort": preview["real_pixel_cohort"],
            "fas_root": preview["fas_root"], "source_manifest": preview["source_manifest"],
            "source_manifest_sha256": MANIFEST_SHA, "preprocessing_sha256": preview["preprocessing_sha256"],
            "msu_reader_sha256": preview["msu_reader_sha256"],
            "bootstrap": {"repetitions": 2000, "seed": addressed_seed(SEED, "MSU-MFSD", "crop-ablation-bootstrap")},
            "conservative_estimate_bytes": estimate,
            "implementation_sha256": {str(p.relative_to(project)): digest(p) for directory in ("src", "scripts")
                                      for p in sorted((project / directory).rglob("*.py"))}}
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def checked_inputs(path: Path, sha: str) -> dict[str, Any]:
    require_private_output(path)
    plan = read_json(path, sha)
    if (plan["scope"] != "MSU_frozen_matched_crop_development_diagnostic"
            or plan["authority"] != AUTHORITY or plan["budget"] != BUDGET or plan["policy"] != REAL_PIXEL_POLICY
            or plan["config"]["model"]["variant"] != "pretrained_spatial"
            or plan["config"]["scoring"] != SCORING or plan["config"]["seed"] != SEED):
        raise PermissionError("fixed frozen-model crop-only authority and scoring required")
    project = Path(__file__).resolve().parents[2]
    for name, expected in plan["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("ablation implementation changed")
    root = Path(plan["preview_root"])
    read_json(root / "visual-approval.json", plan["approval_sha256"])
    preview, selected = approved_preview(root)
    entry_videos = [item_video(e) for e in plan["entries"]]
    validate_population(entry_videos)
    if (entry_videos != selected
            or plan["cohort"] != preview["real_pixel_cohort"]):
        raise ValueError("ablation must retain every eligible validation original")
    diagnostic = read_json(Path(preview["diagnostic_root"]) / "plan.json", preview["diagnostic_plan_sha256"])
    expected_config = copy.deepcopy(diagnostic["config"])
    expected_config["model"]["variant"] = "pretrained_spatial"
    expected_config["execution"] = {name: name in {"source_decode", "source_scoring"}
                                    for name in expected_config["execution"]}
    expected_config["budget"] = {"max_seconds": BUDGET["max_seconds"],
                                 "max_output_bytes": BUDGET["max_output_bytes"], "max_optimizer_steps": None}
    if plan["config"] != expected_config:
        raise ValueError("ablation model/config lineage drift")
    for name in ("fas_root", "source_manifest", "preprocessing_sha256", "msu_reader_sha256"):
        if plan[name] != preview[name]:
            raise ValueError("ablation decoder lineage drift")
    if plan["source_manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("ablation source manifest drift")
    if plan["bootstrap"] != {"repetitions": 2000, "seed": addressed_seed(SEED, "MSU-MFSD", "crop-ablation-bootstrap")}:
        raise ValueError("fixed paired-bootstrap recipe required")
    rows = {f"{r['dataset']}/{r['video_id']}": r
            for r in read_json(Path(plan["source_manifest"]), MANIFEST_SHA)["rows"]}
    for entry, video in zip(plan["entries"], selected, strict=True):
        if (entry["samples"] != full_middle_samples(video)
                or any(entry[k] != value for k, value in source_media_entry(video, rows[video.key]).items())):
            raise ValueError("ablation native-window/archive binding drift")
    return plan


def render_pair(video: Video, samples: list[dict[str, Any]], frames: dict[int, np.ndarray],
                legacy_module) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(samples) != 16 or [s["clip_slot"] for s in samples] != list(range(16)):
        raise ValueError("exact fixed T16 sample order required")
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in ARMS}
    margins, native_hashes = [], []
    with np.load(video.cache_path, allow_pickle=False) as cache:
        old = cache["frames"]
        for sample in samples:
            rgb = frames[sample["native_order"]]
            points = np.asarray(sample["landmarks"])
            legacy, _ = legacy_module.shared_context_crop(rgb, np.asarray(sample["box"]), points)
            legacy = cv2.resize(legacy, (256, 256), interpolation=cv2.INTER_CUBIC)
            if not np.array_equal(legacy, old[sample["cache_index"]]):
                raise ValueError("ablation legacy crop differs from locked cache")
            corrected, geometry = crop_real_pixels(rgb, points)
            outputs["legacy"].append(legacy)
            outputs["real_pixels"].append(corrected)
            margins.append(geometry["minimum_source_margin_pixels"])
            native_hashes.append(video.frame_hashes[sample["native_order"]])
    return {name: np.stack(images) for name, images in outputs.items()}, {
        "legacy_cache_pixel_matches": 16, "minimum_source_margin_pixels": min(margins),
        "padding_fraction": 0.0, "bilinear_support_verified_frames": 16,
        "native_RGB_sha256": native_hashes}


def score_pair(model, pair: dict[str, np.ndarray], key: str, config: dict[str, Any],
               device: torch.device) -> dict[str, list[float]]:
    if set(pair) != set(ARMS):
        raise ValueError("both fixed crop arms required")
    profiles = {}
    for name in ARMS:
        frames = pair[name]
        if frames.dtype != np.uint8 or frames.shape != (16, 256, 256, 3):
            raise ValueError("sixteen uint8 RGB256 frames per arm required")
        clip = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)
        measured = residual_profile(model, clip[None].to(device), [key], seed=config["seed"],
                                    **config["scoring"])[0].cpu()
        profiles[name] = measured.tolist()
    return profiles


def compare_predictions(arms: dict[str, list[Prediction]], bootstrap: dict[str, int]) -> dict[str, Any]:
    if tuple(arms) != ARMS:
        raise ValueError("fixed ordered crop arms required")
    summary = {name: auroc_summary(rows) for name, rows in arms.items()}
    values = [summary[name]["auroc_usable"] for name in ARMS]
    return {"arms": summary, "real_pixels_minus_legacy": values[1] - values[0]
            if all(v is not None for v in values) else None,
            "paired_subject_bootstrap": paired_auroc_bootstrap(arms, **bootstrap),
            "grouping_unit": "declared_subject_group", "small_group_count_warning": True}


def run_ablation(path: Path, sha: str) -> None:
    plan = checked_inputs(path, sha)
    root, fas_root = path.parent, Path(plan["fas_root"])
    if (root / "started.json").exists():
        raise ValueError("ablation already attempted; no automatic retry")
    write_json(root / "started.json", {"plan_sha256": sha})
    deadline = time.monotonic() + BUDGET["max_seconds"] - 20
    for folder in ("records", "staging", "pairs"):
        (root / folder).mkdir()
    cv2.setNumThreads(1)
    if (digest(fas_root / "src/fas/preprocessing.py") != plan["preprocessing_sha256"]
            or digest(fas_root / "scripts/inventory_msu.py") != plan["msu_reader_sha256"]):
        raise ValueError("bound source code changed")
    sys.path.insert(0, str(fas_root / "src"))
    legacy_module = importlib.import_module("fas.preprocessing")
    if Path(legacy_module.__file__).resolve() != (fas_root / "src/fas/preprocessing.py").resolve():
        raise ValueError("unbound legacy preprocessing import")
    decoders, helper_sha = decoder_policy(plan["config"])
    if decoders["MSU-MFSD"] != "fas_strict_ffmpeg" or helper_sha is None:
        raise ValueError("strict MSU native replay required; no fallback")
    native = FasNativeReplay(fas_root, helper_sha, Path(plan["source_manifest"]), MANIFEST_SHA)
    with ExitStack() as stack:
        context: dict[str, Any] = {}
        for index, entry in enumerate(plan["entries"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("ablation shared wall cap")
            video = item_video(entry)
            staged = root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}"
            write_json(root / "records" / f"started-{index:02d}.json", {"index": index})
            evidence, failure, pair_sha = None, None, None
            try:
                stage_member(entry, fas_root, fas_root, staged, stack, context, deadline,
                             allow_validation_attacks=True)
                replay = native.bind(replace(video, media_path=str(staged)),
                                     root / "native-replay" / f"original-{index:02d}",
                                     expected_receipt_sha256=entry["receipt_sha256"])
                frames = read_native_frames(video, staged, {s["native_order"] for s in entry["samples"]},
                                            min(deadline, time.monotonic() + 180), replay)
                pair, evidence = render_pair(video, entry["samples"], frames, legacy_module)
                cache_path = root / "pairs" / f"{index:02d}.npz"
                with cache_path.open("xb") as stream:
                    np.savez(stream, **pair)
                pair_sha = digest(cache_path)
                del pair, frames
            except MediaFailure as error:
                failure = str(error)
            finally:
                staged.unlink(missing_ok=True)
            write_json(root / "records" / f"crop-{index:02d}.json",
                       {"key": video.key, "failure": failure, "evidence": evidence, "pair_sha256": pair_sha})
            if tree_bytes(root) > BUDGET["max_output_bytes"]:
                raise OSError("ablation output cap")
            print({"phase": "crop", "originals": index + 1, "total": 22,
                   "status": failure or "prepared"}, flush=True)
    write_json(root / "device.json", configure_gpu(plan))
    model = build_model(plan["config"]).cuda().eval()
    model.backbone_frame_batch = 4
    if model.adapter is not None or any(p.requires_grad for p in model.parameters()):
        raise ValueError("frozen backbone without adapters required")
    frozen_sha = tensor_digest(model.backbone.state_dict())
    torch.cuda.reset_peak_memory_stats()
    arms: dict[str, list[Prediction]] = {name: [] for name in ARMS}
    for index, entry in enumerate(plan["entries"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("ablation shared wall cap")
        video = item_video(entry)
        crop = read_json(root / "records" / f"crop-{index:02d}.json")
        if crop["key"] != video.key:
            raise ValueError("ablation crop receipt identity drift")
        profiles = None
        if crop["failure"] is None:
            pair_path = root / "pairs" / f"{index:02d}.npz"
            if digest(pair_path) != crop["pair_sha256"]:
                raise ValueError("ablation crop-pair cache changed")
            with np.load(pair_path, allow_pickle=False) as cache:
                pair = {name: cache[name] for name in ARMS}
            profiles = score_pair(model, pair, video.key, plan["config"], torch.device("cuda"))
            del pair
        rows = {}
        for name in ARMS:
            profile = profiles[name] if profiles is not None else None
            score = float(np.asarray(profile, dtype=np.float64).mean()) if profile is not None else None
            prediction = Prediction(video.key, video.dataset, video.group, video.partition,
                                    video.label, score, crop["failure"], video.attack_family)
            arms[name].append(prediction)
            rows[name] = {**asdict(prediction), "profile": profile}
        write_json(root / "records" / f"score-{index:02d}.json",
                   {"plan_sha256": sha, "crop_receipt_sha256": digest(root / "records" / f"crop-{index:02d}.json"),
                    "arms": rows})
        print({"phase": "scoring", "originals": index + 1, "total": 22}, flush=True)
    if tensor_digest(model.backbone.state_dict()) != frozen_sha:
        raise ValueError("frozen DDPM state changed during ablation")
    comparison = compare_predictions(arms, plan["bootstrap"])
    write_json(root / "result.json", {
        "scope": plan["scope"], "plan_sha256": sha, "comparison": comparison,
        "policy": REAL_PIXEL_POLICY, "authority": AUTHORITY, "budget": BUDGET,
        "originals": 22, "model_scoring_profiles": sum(r.score is not None for rows in arms.values() for r in rows),
        "training_runs": 0, "threshold_fitting_runs": 0,
        "fitting_pixel_access": 0, "calibration_pixel_access": 0, "target_pixel_access": 0,
        "frozen_backbone_sha256_before_and_after": frozen_sha,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "records_sha256": {p.name: digest(p) for p in sorted((root / "records").glob("*.json"))},
    })


def summarize(path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    root = path.parent
    plan = read_json(path)
    for index, entry in enumerate(plan["entries"]):
        (root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}").unlink(missing_ok=True)
    result = read_json(root / "result.json") if (root / "result.json").exists() else None
    if result is not None:
        if result["plan_sha256"] != digest(path):
            raise ValueError("ablation result plan drift")
        for name, expected in result["records_sha256"].items():
            if digest(root / "records" / name) != expected:
                raise ValueError("ablation result receipt drift")
    public = None if result is None else {k: v for k, v in result.items() if k != "records_sha256"}
    return {"completed": bool(supervision["completed"] and result is not None),
            "supervision": supervision, "plan_sha256": digest(path), "result": public,
            "visual_review": "accepted", "development_subset_not_full_MSU_benchmark": True,
            "result_sha256": digest(root / "result.json") if result is not None else None}
