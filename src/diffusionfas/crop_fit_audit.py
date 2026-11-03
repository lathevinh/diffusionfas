"""Metadata-only no-padding crop coverage, not model or cohort selection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any

import numpy as np

from .celebahq_geometry import POLICY, square_geometry
from .data import Video, load_manifest
from .fas_import import MANIFEST_SHA, read_media_receipt
from .io import digest, object_digest, read_json, require_private_output, write_json
from .preprocessing import MediaFailure
from .source_preparation import selected_sources, source_media_entry
from .source_probe import SOURCES

BUDGET = {"max_seconds": 600, "max_rss_bytes": 2 * 1024**3, "max_output_bytes": 64 * 1024**2}
CRITERION = {
    "name": "celebahq_square_real_pixels_only_v1",
    "bounds": "all_four_source_quad_corners_inside_[0,width-1]x[0,height-1]",
    "padding_allowed": False, "crop_shrinking_allowed": False, "crop_shifting_allowed": False,
    "validation_window": "existing_lower_middle_T16",
    "fitting_primary": "all_existing_8fps_cache_frames_fit",
    "alternate_windows_are_diagnostic_only": True,
    "dataset_or_original_selection_authorized": False,
}


def quad_margin(landmarks: np.ndarray, width: int, height: int) -> float:
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        raise ValueError("positive native dimensions required")
    quad = np.asarray(square_geometry(landmarks)["quad"])
    return float(np.min(np.stack((quad[:, 0], quad[:, 1],
                                 width - 1 - quad[:, 0], height - 1 - quad[:, 1]))))


def window_counts(valid: np.ndarray) -> tuple[int, int]:
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError("one-dimensional boolean frame eligibility required")
    if len(valid) < 16:
        raise ValueError("prepared source cache must contain at least sixteen frames")
    totals = np.convolve(valid.astype(np.int64), np.ones(16, dtype=np.int64), mode="valid")
    return int(np.sum(totals == 16)), len(totals)


def audit_cache(video: Video, native_dimensions: list[tuple[int, int]]) -> dict[str, Any]:
    if not video.cache_path or digest(Path(video.cache_path)) != video.cache_sha256:
        raise ValueError("audit cache commitment mismatch")
    if len(native_dimensions) != len(video.timestamps):
        raise ValueError("native dimension population mismatch")
    with np.load(video.cache_path, allow_pickle=False) as cache:
        timestamps, native = cache["timestamps"], cache["native_timestamps"]
        orders, landmarks = cache["native_orders"], cache["landmarks"]
        count = len(timestamps)
        if (count < 16 or native.shape != (count,) or orders.shape != (count,)
                or landmarks.shape != (count, 5, 2) or orders.dtype.kind not in "iu"
                or np.any(orders < 0) or np.any(orders >= len(video.timestamps))
                or np.any(np.diff(orders) <= 0) or not np.isfinite(timestamps).all()
                or not np.allclose(np.diff(timestamps), 1 / 8, rtol=0, atol=1e-7)
                or not np.array_equal(native, np.asarray(video.timestamps)[orders])):
            raise ValueError("audit requires intact native-addressed 8fps cache metadata")
        margins, failures = [], Counter()
        for points, order in zip(landmarks, orders):
            try:
                margins.append(quad_margin(points, *native_dimensions[int(order)]))
            except MediaFailure as error:
                failures[str(error)] += 1
                margins.append(None)
    valid = np.array([m is not None and m >= 0 for m in margins], dtype=np.bool_)
    start = (count - 16) // 2
    valid_windows, windows = window_counts(valid)
    finite_margins = [m for m in margins if m is not None]
    return {
        "cached_frames": count, "fit_frames": int(valid.sum()),
        "all_cached_frames_fit": bool(valid.all()), "middle_T16_fit": bool(valid[start:start + 16].all()),
        "valid_T16_windows": valid_windows, "all_T16_windows": windows,
        "has_any_valid_T16_window": valid_windows > 0,
        "minimum_source_margin_pixels": min(finite_margins) if finite_margins else None,
        "geometry_failures": dict(failures), "cache_sha256": video.cache_sha256,
        "image_arrays_decoded": 0,
    }


def population(bonafide_manifest: Path, diagnostic_manifest: Path) -> list[Video]:
    base, _ = load_manifest(bonafide_manifest)
    diagnostic, _ = load_manifest(diagnostic_manifest)
    bonafide = selected_sources(base)
    by_key = {v.key: v for v in bonafide}
    for video in diagnostic:
        if video.dataset not in SOURCES or video.partition != "validation":
            raise ValueError("completed diagnostic must be source-validation only")
        if video.label == "bona_fide" and video != by_key.get(video.key):
            raise ValueError("bonafide lineage differs between preparation and diagnostic")
    attacks = [v for v in diagnostic if v.label == "attack"]
    result = sorted(bonafide + attacks, key=lambda v: (v.dataset, v.partition, v.label, v.key))
    if len({v.key for v in result}) != len(result):
        raise ValueError("audit original identities must be unique")
    return result


def build_plan(bonafide_manifest: Path, diagnostic_root: Path, output: Path) -> Path:
    require_private_output(output / "plan.json")
    if output.exists():
        raise ValueError("new audit output root required")
    previous = read_json(diagnostic_root / "plan.json")
    completion = read_json(diagnostic_root / "completion.json")
    supervision = read_json(diagnostic_root / "supervision.json")
    diagnostic_manifest = diagnostic_root / "diagnostic-manifest.json"
    prior_sha = digest(diagnostic_root / "plan.json")
    if (previous["scope"] != "source_validation_development_diagnostic"
            or not completion["completed"] or not supervision["completed"]
            or completion["plan_sha256"] != prior_sha or supervision["plan_sha256"] != prior_sha
            or digest(bonafide_manifest) != previous["manifest_sha256"]
            or digest(diagnostic_manifest) != completion["manifest_sha256"]
            or previous["source_manifest_sha256"] != MANIFEST_SHA):
        raise ValueError("audit requires completed, unchanged source preparation and diagnostic")
    videos = population(bonafide_manifest, diagnostic_manifest)
    source_path = Path(previous["source_manifest"])
    read_json(source_path, MANIFEST_SHA)
    project = Path(__file__).resolve().parents[2]
    plan = {"scope": "source_metadata_no_padding_coverage", "budget": BUDGET, "criterion": CRITERION,
            "geometry": POLICY, "bonafide_manifest": str(bonafide_manifest.resolve()),
            "bonafide_manifest_sha256": digest(bonafide_manifest),
            "diagnostic_manifest": str(diagnostic_manifest.resolve()),
            "diagnostic_manifest_sha256": digest(diagnostic_manifest),
            "source_manifest": str(source_path), "source_manifest_sha256": MANIFEST_SHA,
            "originals": len(videos), "population_sha256": object_digest([v.to_dict() for v in videos]),
            "implementation_sha256": {str(p.relative_to(project)): digest(p) for directory in ("src", "scripts")
                                      for p in sorted((project / directory).rglob("*.py"))}}
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def run_audit(path: Path, sha: str) -> None:
    plan = read_json(path, sha)
    require_private_output(path)
    if (plan["scope"] != "source_metadata_no_padding_coverage" or plan["budget"] != BUDGET
            or plan["criterion"] != CRITERION or plan["geometry"] != POLICY
            or plan["source_manifest_sha256"] != MANIFEST_SHA):
        raise PermissionError("fixed no-padding metadata-only audit required")
    root = path.parent
    if (root / "started.json").exists():
        raise ValueError("audit already attempted; no automatic retry")
    project = Path(__file__).resolve().parents[2]
    for name, expected in plan["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("audit implementation changed")
    for name in ("bonafide_manifest", "diagnostic_manifest"):
        read_json(Path(plan[name]), plan[f"{name}_sha256"])
    videos = population(Path(plan["bonafide_manifest"]), Path(plan["diagnostic_manifest"]))
    if object_digest([v.to_dict() for v in videos]) != plan["population_sha256"]:
        raise ValueError("audit population changed")
    rows = {f"{r['dataset']}/{r['video_id']}": r
            for r in read_json(Path(plan["source_manifest"]), MANIFEST_SHA)["rows"]}
    write_json(root / "started.json", {"plan_sha256": sha})
    deadline = time.monotonic() + BUDGET["max_seconds"] - 10
    groups, records = {}, []
    for video in videos:
        if time.monotonic() >= deadline:
            raise TimeoutError("metadata audit wall cap")
        name = f"{video.dataset}:{video.partition}:{video.label}"
        group = groups.setdefault(name, Counter())
        group["originals"] += 1
        record = {"key": video.key, "dataset": video.dataset, "partition": video.partition,
                  "label": video.label, "group": video.group, "status": video.status}
        if video.status != "prepared":
            group["unassessed_existing_failure"] += 1
            record["geometry"] = None
        else:
            entry = source_media_entry(video, rows[video.key])
            receipt, receipt_sha = read_media_receipt(rows[video.key])
            if receipt_sha != entry["receipt_sha256"] or receipt["binary_label"] != video.label:
                raise ValueError("audit media receipt binding drift")
            dimensions = [(f["width"], f["height"]) for f in receipt["frame_index"]]
            measured = audit_cache(video, dimensions)
            record["geometry"] = measured
            group["assessed_caches"] += 1
            for key in ("all_cached_frames_fit", "middle_T16_fit", "has_any_valid_T16_window"):
                group[key] += int(measured[key])
            for key in ("cached_frames", "fit_frames", "valid_T16_windows", "all_T16_windows"):
                group[key] += measured[key]
            group["invalid_landmark_frames"] += sum(measured["geometry_failures"].values())
        records.append(record)
    write_json(root / "private-results.json", {"plan_sha256": sha, "records": records})
    write_json(root / "result.json", {
        "scope": plan["scope"], "plan_sha256": sha, "criterion": CRITERION,
        "originals": len(videos), "groups": groups,
        "source_video_decodes": 0, "image_arrays_decoded": 0, "model_scoring_runs": 0,
        "training_runs": 0, "calibration_pixel_access": 0, "target_pixel_access": 0,
        "dataset_or_original_selection_performed": False,
        "unknown_geometry_is_not_a_crop_failure": True,
        "private_results_sha256": digest(root / "private-results.json"),
    })
