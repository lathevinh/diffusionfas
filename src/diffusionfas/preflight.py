"""Aggregate-only metadata and runtime preflight."""

from __future__ import annotations

from collections import Counter, defaultdict
import importlib.metadata
from pathlib import Path
import platform
import shutil
from typing import Any

from .data import DOMAINS, Video, clip_start, window_indices


def inventory(videos: list[Video]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for dataset in DOMAINS:
        for partition in ("fitting", "validation", "calibration", "evaluation"):
            for label in ("bona_fide", "attack"):
                subset = [v for v in videos if (v.dataset, v.partition, v.label) == (dataset, partition, label)]
                eligible = {}
                reasons: dict[str, dict[str, int]] = {}
                for duration in (1, 2, 3):
                    success, failures = 0, Counter()
                    for video in subset:
                        if video.status != "eligible_pending_tracking" and video.status != "prepared":
                            failures[video.status] += 1
                            continue
                        try:
                            start = clip_start(video, duration)
                            window_indices(video.timestamps, start, duration)
                        except ValueError as error:
                            failures[str(error)] += 1
                        else:
                            success += 1
                    eligible[str(duration)] = success
                    reasons[str(duration)] = dict(failures)
                buckets[f"{dataset}/{partition}/{label}"] = {
                    "originals": len(subset), "groups": len({v.group for v in subset}),
                    "known_subjects": len({v.subject for v in subset if v.subject}),
                    "unknown_subject_originals": sum(v.subject is None for v in subset),
                    "eligible_by_seconds_before_new_tracking": eligible,
                    "ineligible_reasons_by_seconds": reasons,
                    "extracted_media_paths": sum(v.media_path is not None for v in subset),
                    "prepared_caches": sum(v.cache_path is not None for v in subset),
                }
    folds = {}
    strict_blockers = {}
    for target in DOMAINS:
        support = defaultdict(dict)
        for source in DOMAINS:
            if source != target:
                for role in ("fitting", "validation", "calibration"):
                    item = buckets[f"{source}/{role}/bona_fide"]
                    support[source][role] = item["eligible_by_seconds_before_new_tracking"]["2"]
        folds[target] = dict(support)
        strict_blockers[target] = [
            source for source, roles in support.items() if roles["calibration"] < 20
        ]
    return {
        "schema_version": 1, "scope": "saved_metadata_only_not_new_clip_acceptance",
        "populations": buckets, "fold_bonafide_2s_support_before_tracking": folds,
        "strict_bpcer_5pct_insufficient_calibration_sources": strict_blockers,
        "originals": len(videos), "new_media_decodes": 0, "new_training_runs": 0,
        "new_target_evaluations": 0, "checkpoint_downloads": 0,
    }


def resources(storage_root: Path) -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("torch", "numpy", "opencv-python-headless", "diffusers", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            devices.append({"index": index, "name": properties.name,
                            "total_bytes": total, "free_bytes": free})
    return {
        "python": platform.python_version(), "packages": packages, "cuda": devices,
        "disk_free_bytes": shutil.disk_usage(storage_root).free,
        "cpu_count": __import__("os").cpu_count(),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffprobe_available": shutil.which("ffprobe") is not None,
        "scientific_execution_authorized": False,
    }
