"""Immutable source-only preparation batches with per-original accounting."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from fractions import Fraction
import math
from pathlib import Path
import time
from typing import Any

from .data import Video, addressed_seed, clip_start, load_manifest, window_indices
from .fas_import import MANIFEST_SHA, read_media_receipt
from .io import digest, object_digest, read_json, require_private_output, write_json
from .native_replay import decoder_policy
from .source_probe import SOURCES, stage_member
from .supervision import tree_bytes

DETECTOR_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
OPERATIONS = ("source_decode", "target_decode", "source_training", "source_scoring",
              "target_evaluation", "checkpoint_download")


def selected_sources(videos: list[Video]) -> list[Video]:
    return sorted(
        (v for v in videos if v.dataset in SOURCES and v.label == "bona_fide"
         and v.partition in {"fitting", "validation"}),
        key=lambda v: (v.dataset, v.partition, v.key),
    )


def build_plan(
    metadata: Path, source_manifest: Path, template: Path, fas_root: Path,
    detector: Path, output: Path, *, max_seconds: float, max_rss_bytes: int,
    max_output_bytes: int, source_manifest_sha: str = MANIFEST_SHA,
    detector_sha: str = DETECTOR_SHA,
) -> Path:
    require_private_output(output / "placeholder")
    if output.exists():
        raise ValueError("new preparation root required")
    if not math.isfinite(max_seconds) or min(max_seconds, max_rss_bytes, max_output_bytes) <= 0:
        raise ValueError("positive approved preparation caps required")
    videos, provenance = load_manifest(metadata)
    if provenance["source_manifest_sha256"] != source_manifest_sha:
        raise ValueError("source population differs from imported provenance")
    rows = {f"{r['dataset']}/{r['video_id']}": r
            for r in read_json(source_manifest, source_manifest_sha)["rows"]}
    config = read_json(template)
    if config["target"] != "OULU-NPU" or config["clip"]["fps"] != 8 or config["clip"]["duration"] != 2:
        raise ValueError("this batch is fixed to OULU-held-out 2s/8FPS source preparation")
    config["execution"] = {name: name == "source_decode" for name in OPERATIONS}
    config["budget"] = {"max_seconds": max_seconds, "max_output_bytes": max_output_bytes,
                        "max_optimizer_steps": None}
    decoder_policy(config)
    if digest(detector) != detector_sha:
        raise ValueError("approved detector binding mismatch")
    selected, entries = selected_sources(videos), []
    for video in selected:
        row = rows.get(video.key)
        if row is None or row["binary_label"] != video.label:
            raise ValueError("source original identity/label differs from accepted population")
        failure = None
        grid_frames = 0
        try:
            if video.status != "eligible_pending_tracking":
                raise ValueError(video.status)
            window_indices(video.timestamps, clip_start(video, 2))
            interval = video.interval
            if interval is None:
                raise ValueError("insufficient_duration")
            duration = int((interval[1] - interval[0] + 1e-8) * 8) / 8
            grid_frames = len(window_indices(video.timestamps, interval[0], duration, 8))
        except ValueError as error:
            failure = str(error)
        entry: dict[str, Any] = {"video": video.to_dict(), "metadata_failure": failure}
        if failure is None:
            receipt, receipt_sha = read_media_receipt(row)
            if (
                receipt["media_sha256"] != video.media_sha256
                or receipt["decoded_rgb_sequence_sha256"] != video.decoded_sha256
                or tuple(f["rgb_sha256"] for f in receipt["frame_index"]) != video.frame_hashes
                or tuple(float(Fraction(f["timestamp_seconds"])) for f in receipt["frame_index"]) != video.timestamps
            ):
                raise ValueError("preparation receipt differs from imported evidence")
            size = receipt["byte_size"]
            if not 0 < size <= 128 * 1024**2:
                raise ValueError("source original exceeds existing exact-member staging cap")
            entry.update(member=receipt["media_relpath"], byte_size=size, receipt_sha256=receipt_sha,
                         grid_frames=grid_frames)
        entries.append(entry)
    if not entries:
        raise ValueError("nonempty source fitting/validation population required")
    raw_bytes = sum(e.get("grid_frames", 0) for e in entries) * 256 * 256 * 3
    expected_bytes = raw_bytes + max(e.get("byte_size", 0) for e in entries) + 2 * metadata.stat().st_size
    if expected_bytes + 64 * 1024**2 > max_output_bytes:
        raise ValueError("conservative cache/staging/manifest estimate exceeds approved output cap")
    project = Path(__file__).resolve().parents[2]
    implementation = {str(p.relative_to(project)): digest(p)
                      for directory in ("src", "scripts") for p in sorted((project / directory).rglob("*.py"))}
    plan = {
        "schema_version": 1, "scope": "source_bonafide_fitting_validation_only",
        "metadata": str(metadata.resolve()), "metadata_sha256": digest(metadata),
        "source_manifest": str(source_manifest.resolve()), "source_manifest_sha256": source_manifest_sha,
        "fas_root": str(fas_root.resolve()), "detector": str(detector.resolve()),
        "detector_sha256": detector_sha,
        "preprocessing_sha256": digest(fas_root / "src/fas/preprocessing.py"),
        "msu_reader_sha256": digest(fas_root / "scripts/inventory_msu.py"),
        "implementation_sha256": implementation, "entries": entries,
        "budget": {"max_seconds": max_seconds, "max_rss_bytes": max_rss_bytes,
                   "max_output_bytes": max_output_bytes},
        "uncompressed_RGB_estimate_bytes": raw_bytes, "conservative_estimate_bytes": expected_bytes,
        "config": config,
    }
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def item_video(item: dict[str, Any]) -> Video:
    values = dict(item["video"])
    values["timestamps"], values["frame_hashes"] = tuple(values["timestamps"]), tuple(values["frame_hashes"])
    return Video(**values)


def run_preparation(plan_path: Path, plan_sha: str) -> None:
    import cv2
    import torch

    from .dataset import ClipCache
    from .native_replay import FasNativeReplay
    from .preprocessing import FasFaceProvider, MediaFailure, prepare_video

    plan = read_json(plan_path, plan_sha)
    root = plan_path.parent
    require_private_output(root / "placeholder")
    if (root / "records").exists():
        raise ValueError("batch already attempted; no automatic retry/resume")
    project = Path(__file__).resolve().parents[2]
    for name, expected in plan["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("preparation implementation changed after approval plan")
    if digest(Path(plan["metadata"])) != plan["metadata_sha256"]:
        raise ValueError("imported metadata changed after selection")
    if digest(Path(plan["source_manifest"])) != plan["source_manifest_sha256"]:
        raise ValueError("source manifest changed after selection")
    videos, provenance = load_manifest(Path(plan["metadata"]))
    expected = [v.to_dict() for v in selected_sources(videos)]
    if object_digest(expected) != object_digest([e["video"] for e in plan["entries"]]):
        raise ValueError("preparation population changed")
    config = plan["config"]
    if (config["target"] != "OULU-NPU" or not config["execution"]["source_decode"]
            or any(config["execution"][name] for name in OPERATIONS if name != "source_decode")):
        raise PermissionError("only source preparation is authorized")
    if provenance["source_manifest_sha256"] != plan["source_manifest_sha256"]:
        raise ValueError("source provenance binding changed")
    fas_root, detector = Path(plan["fas_root"]), Path(plan["detector"])
    if digest(fas_root / "scripts/inventory_msu.py") != plan["msu_reader_sha256"]:
        raise ValueError("source archive reader changed after plan")
    decoders, helper_sha = decoder_policy(config)
    native = None
    if any(decoders[e["video"]["dataset"]] == "fas_strict_ffmpeg" and not e["metadata_failure"]
           for e in plan["entries"]):
        if helper_sha is None:
            raise ValueError("explicit native decoder helper hash required")
        native = FasNativeReplay(fas_root, helper_sha, Path(plan["source_manifest"]),
                                 plan["source_manifest_sha256"])
    started = time.monotonic()
    deadline = started + plan["budget"]["max_seconds"] - 10
    provider = FasFaceProvider(fas_root, plan["preprocessing_sha256"], detector, plan["detector_sha256"])
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    cache_reader = ClipCache(256)
    staging, records, caches = root / "staging", root / "records", root / "caches"
    for directory in (staging, records, caches):
        directory.mkdir()
    completed: dict[str, Video] = {}
    with ExitStack() as stack:
        msu_context: dict[str, Any] = {}
        for index, item in enumerate(plan["entries"]):
            video = item_video(item)
            before = time.monotonic()
            if before >= deadline:
                raise TimeoutError("approved source preparation wall cap")
            remaining_bytes = plan["budget"]["max_output_bytes"] - tree_bytes(root)
            if remaining_bytes <= 0:
                raise OSError("approved source preparation output cap")
            failure = item["metadata_failure"]
            result = replace(video, status=f"preparation_failure:{failure}") if failure else video
            timings = {"stage_seconds": 0.0, "preparation_seconds": 0.0, "verification_seconds": 0.0}
            grid_frames, windows = 0, 0
            if not failure:
                write_json(records / f"started-{index:06d}.json", {"index": index})
                staged = staging / f"source-{index:06d}{Path(item['member']).suffix}"
                try:
                    stage_member(item, fas_root, fas_root, staged, stack, msu_context, deadline)
                    timings["stage_seconds"] = time.monotonic() - before
                    write_json(records / f"staged-{index:06d}.json", {"bytes_verified": item["byte_size"]})
                    bound = replace(video, media_path=str(staged))
                    prefix = root / "native-replay" / f"original-{index:06d}"
                    replay = None
                    if decoders[video.dataset] == "fas_strict_ffmpeg":
                        if native is None:
                            raise ValueError("native decoder not initialized")
                        replay = native.bind(bound, prefix, expected_receipt_sha256=item["receipt_sha256"])
                    prepare_started = time.monotonic()
                    write_json(records / f"prepare-called-{index:06d}.json", {"index": index})
                    try:
                        result = prepare_video(
                            bound, provider, caches / f"{index:06d}.npz", fps=8,
                            max_seconds=min(180, max(0.001, deadline - time.monotonic())),
                            max_output_bytes=min(512 * 1024**2, remaining_bytes),
                            native_replay=replay,
                        )
                    except MediaFailure as error:
                        failure = str(error)
                        result = replace(video, status=f"preparation_failure:{failure}")
                    timings["preparation_seconds"] = time.monotonic() - prepare_started
                finally:
                    staged.unlink(missing_ok=True)
                if result.status == "prepared":
                    verify_started = time.monotonic()
                    for seed in (None, addressed_seed(20261103, video.key, "cache-QA", 0),
                                 addressed_seed(20261103, video.key, "cache-QA", 1)):
                        clip = cache_reader.get(result, seed=seed)
                        if clip.shape != (16, 3, 256, 256) or not torch.isfinite(clip).all():
                            raise ValueError("prepared cache does not yield finite 2s RGB256 windows")
                        windows += 1
                    timings["verification_seconds"] = time.monotonic() - verify_started
                    grid_frames = item["grid_frames"]
            result = replace(result, media_path=None)
            completed[video.key] = result
            write_json(records / f"result-{index:06d}.json", {
                "video": result.to_dict(), "failure": failure, "metadata_failure": item["metadata_failure"],
                "timings": timings, "grid_frames": grid_frames, "verified_cache_windows": windows,
                "wall_seconds": time.monotonic() - before,
            })
            if tree_bytes(root) > plan["budget"]["max_output_bytes"]:
                raise OSError("approved output cap including caches and diagnostics")
            print({"completed_originals": index + 1, "total_originals": len(plan["entries"]),
                   "dataset": video.dataset, "partition": video.partition, "status": result.status,
                   "wall_seconds": time.monotonic() - started}, flush=True)
    updated = [completed.get(v.key, v) for v in videos]
    write_json(root / "prepared-manifest.json", {
        "schema_version": 1, "videos": [v.to_dict() for v in updated],
        "provenance": {**provenance, "parent_manifest_sha256": plan["metadata_sha256"],
                       "preparation_plan_sha256": plan_sha,
                       "decoder_policy": {"decoders": decoders, "native_replay_helper_sha256": helper_sha},
                       "preprocessing_sha256": plan["preprocessing_sha256"],
                       "detector_sha256": plan["detector_sha256"], "clip_config": config["clip"]},
    })
    write_json(root / "completion.json", {"completed": True, "plan_sha256": plan_sha,
                                        "wall_seconds": time.monotonic() - started,
                                        "prepared_manifest_sha256": digest(root / "prepared-manifest.json")})


def summarize_preparation(plan_path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    plan = read_json(plan_path)
    root = plan_path.parent
    counts: dict[str, Counter] = {}
    failures: Counter = Counter()
    timings: Counter = Counter()
    windows, grid_frames, staged_bytes, records_count = 0, 0, 0, 0
    for index, item in enumerate(plan["entries"]):
        bucket = counts.setdefault(f"{item['video']['dataset']}:{item['video']['partition']}", Counter())
        bucket["originals"] += 1
        result_path = root / "records" / f"result-{index:06d}.json"
        if result_path.exists():
            record = read_json(result_path)
            records_count += 1
            if record["video"]["key"] != item["video"]["key"]:
                raise ValueError("per-original receipt identity mismatch")
            state = ("metadata_failure" if item["metadata_failure"] else
                     "prepared" if record["video"]["status"] == "prepared" else "preparation_failure")
            bucket[state] += 1
            if record["failure"]:
                failures[record["failure"]] += 1
            windows += record["verified_cache_windows"]
            grid_frames += record["grid_frames"]
            timings.update(record["timings"])
        elif item["metadata_failure"]:
            bucket["metadata_failure"] += 1
            failures[item["metadata_failure"]] += 1
        elif (root / "records" / f"started-{index:06d}.json").exists():
            bucket["interrupted"] += 1
        else:
            bucket["not_attempted"] += 1
        staged_receipt = root / "records" / f"staged-{index:06d}.json"
        if staged_receipt.exists():
            staged_bytes += read_json(staged_receipt)["bytes_verified"]
        staged = root / "staging" / f"source-{index:06d}{Path(item.get('member', 'unused')).suffix}"
        staged.unlink(missing_ok=True)
    completion = read_json(root / "completion.json") if (root / "completion.json").exists() else None
    if completion is not None and (
        completion["plan_sha256"] != digest(plan_path)
        or completion["prepared_manifest_sha256"] != digest(root / "prepared-manifest.json")
    ):
        raise ValueError("completed manifest/selection binding mismatch")
    return {
        "schema_version": 1,
        "completed": supervision["completed"] and completion is not None and records_count == len(plan["entries"]),
        "scope": plan["scope"], "target": "OULU-NPU", "counts_by_domain_partition": counts,
        "original_denominator": len(plan["entries"]),
        "metadata_eligible_originals": sum(not e["metadata_failure"] for e in plan["entries"]),
        "prepared_originals": sum(c["prepared"] for c in counts.values()),
        "failures_by_reason": failures, "verified_cache_windows": windows,
        "successful_cache_grid_frames": grid_frames, "timings_seconds": timings,
        "raw_original_bytes_staged_and_verified": staged_bytes, "owned_output_bytes": tree_bytes(root),
        "uncompressed_RGB_estimate_bytes": plan["uncompressed_RGB_estimate_bytes"],
        "selection_plan_sha256": digest(plan_path), "input_metadata_sha256": plan["metadata_sha256"],
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "prepared_manifest_sha256": completion["prepared_manifest_sha256"] if completion else None,
        "implementation_sha256": plan["implementation_sha256"], "budget": plan["budget"],
        "supervision": supervision, "decoder_policy": plan["config"]["preprocessing"],
        "worker_count": 1, "face_detector_device": "CPU", "scientific_training_runs": 0,
        "target_video_decodes": 0, "attack_video_decodes": 0, "source_model_scores": 0,
        "selection_replacements": 0, "native_RGB_waivers": 0, "automatic_retries": 0,
        "calibration_policy_changed": False,
    }
