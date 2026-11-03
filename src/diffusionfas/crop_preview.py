"""Private source-only crop review; no model scoring or automatic continuation."""

from __future__ import annotations

import base64
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import html
import importlib
from pathlib import Path
import random
import sys
import time
from typing import Any

import cv2
import numpy as np

from .celebahq_geometry import POLICY, crop_celebahq
from .data import Video, addressed_seed, load_manifest
from .fas_import import MANIFEST_SHA
from .io import digest, object_digest, read_json, require_private_output, write_json
from .native_replay import FasNativeReplay, decoder_policy
from .preprocessing import MediaFailure
from .source_preparation import source_media_entry
from .source_probe import SOURCES, stage_member
from .supervision import tree_bytes

BUDGET = {"max_seconds": 1200, "max_rss_bytes": 8 * 1024**3,
          "max_output_bytes": 512 * 1024**2}
AUTHORITY = {"source_validation_preview": True, "model_scoring": False, "training": False,
             "calibration_pixels": False, "target_pixels": False, "human_review_required": True}
SEED = 20261104
SLOTS = (0, 7, 15)


def select_videos(videos: list[Video]) -> list[Video]:
    selected = []
    for domain in SOURCES:
        for label in ("bona_fide", "attack"):
            candidates = sorted((v for v in videos if (
                v.dataset, v.partition, v.label, v.status
            ) == (domain, "validation", label, "prepared")), key=lambda v: v.key)
            if len(candidates) < 2:
                raise ValueError("preview requires two prepared originals per source/class")
            rng = random.Random(addressed_seed(SEED, domain, label, "crop-preview"))
            selected.extend(rng.sample(candidates, 2))
    if len({v.key for v in selected}) != 12:
        raise ValueError("preview requires twelve unique originals")
    return selected


def cache_samples(video: Video) -> list[dict[str, Any]]:
    if not video.cache_path or digest(Path(video.cache_path)) != video.cache_sha256:
        raise ValueError("preview cache binding mismatch")
    with np.load(video.cache_path, allow_pickle=False) as cache:
        timestamps = cache["timestamps"]
        orders, native = cache["native_orders"], cache["native_timestamps"]
        boxes, landmarks = cache["boxes"], cache["landmarks"]
        count = len(timestamps)
        if (count < 16 or orders.shape != (count,) or native.shape != (count,)
                or boxes.shape != (count, 4) or landmarks.shape != (count, 5, 2)
                or not np.isfinite(timestamps).all()
                or not np.allclose(np.diff(timestamps), 1 / 8, rtol=0, atol=1e-7)
                or orders.dtype.kind not in "iu" or np.any(np.diff(orders) <= 0)
                or np.any(orders < 0) or np.any(orders >= len(video.timestamps))
                or not np.array_equal(native, np.asarray(video.timestamps)[orders])
                or not np.isfinite(boxes).all() or not np.isfinite(landmarks).all()):
            raise ValueError("preview requires intact full-grid tracking metadata")
        start = (count - 16) // 2
        return [{"clip_slot": slot, "cache_index": start + slot,
                 "native_order": int(orders[start + slot]),
                 "native_timestamp": float(native[start + slot]),
                 "box": boxes[start + slot].tolist(), "landmarks": landmarks[start + slot].tolist()}
                for slot in SLOTS]


def build_plan(diagnostic_root: Path, output: Path) -> Path:
    require_private_output(output / "plan.json")
    if output.exists():
        raise ValueError("new preview root required")
    previous_path = diagnostic_root / "plan.json"
    previous = read_json(previous_path)
    completion = read_json(diagnostic_root / "completion.json")
    supervision = read_json(diagnostic_root / "supervision.json")
    manifest = diagnostic_root / "diagnostic-manifest.json"
    if (previous["scope"] != "source_validation_development_diagnostic"
            or not completion["completed"] or not supervision["completed"]
            or completion["plan_sha256"] != digest(previous_path)
            or supervision["plan_sha256"] != digest(previous_path)
            or digest(manifest) != completion["manifest_sha256"]
            or previous["source_manifest_sha256"] != MANIFEST_SHA):
        raise ValueError("completed pinned source diagnostic required")
    videos, provenance = load_manifest(manifest)
    if (provenance["plan_sha256"] != digest(previous_path)
            or any(v.dataset not in SOURCES or v.partition != "validation" for v in videos)):
        raise ValueError("preview population must be source validation only")
    source_path = Path(previous["source_manifest"])
    rows = {f"{r['dataset']}/{r['video_id']}": r for r in read_json(source_path, MANIFEST_SHA)["rows"]}
    selected = []
    for video in select_videos(videos):
        selected.append({"video": video.to_dict(), **source_media_entry(video, rows[video.key]),
                         "samples": cache_samples(video)})
    png_bound = 36 * (640 * 640 * 3 + 65536) + 72 * (256 * 256 * 3 + 65536)
    estimate = (png_bound + (4 * png_bound + 2) // 3 + max(e["byte_size"] for e in selected)
                + 4 * 16 * 1024**2 + 64 * 1024**2)
    if estimate > BUDGET["max_output_bytes"]:
        raise ValueError("preview conservative output estimate exceeds approval")
    project = Path(__file__).resolve().parents[2]
    plan = {"scope": "private_source_crop_visual_review", "authority": AUTHORITY, "budget": BUDGET,
            "policy": POLICY, "seed": SEED, "slots": list(SLOTS), "selected": selected,
            "diagnostic_root": str(diagnostic_root.resolve()), "diagnostic_plan_sha256": digest(previous_path),
            "diagnostic_completion_sha256": digest(diagnostic_root / "completion.json"),
            "manifest": str(manifest.resolve()), "manifest_sha256": digest(manifest),
            "source_manifest": str(source_path), "source_manifest_sha256": MANIFEST_SHA,
            "fas_root": previous["fas_root"], "preprocessing_sha256": previous["preprocessing_sha256"],
            "msu_reader_sha256": previous["msu_reader_sha256"],
            "decoder_config": {"preprocessing": previous["config"]["preprocessing"]},
            "pool_originals": len(videos), "pool_prepared": sum(v.status == "prepared" for v in videos),
            "conservative_estimate_bytes": estimate,
            "implementation_sha256": {str(p.relative_to(project)): digest(p) for directory in ("src", "scripts")
                                      for p in sorted((project / directory).rglob("*.py"))}}
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def checked_inputs(path: Path, sha: str) -> dict[str, Any]:
    require_private_output(path)
    plan = read_json(path, sha)
    if (plan["authority"] != AUTHORITY or plan["budget"] != BUDGET
            or plan["scope"] != "private_source_crop_visual_review" or plan["policy"] != POLICY
            or plan["seed"] != SEED or plan["slots"] != list(SLOTS)):
        raise PermissionError("fixed preview-only authority, geometry, sample and caps required")
    project = Path(__file__).resolve().parents[2]
    for name, expected in plan["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("preview implementation changed")
    previous_root = Path(plan["diagnostic_root"])
    previous = read_json(previous_root / "plan.json", plan["diagnostic_plan_sha256"])
    completion = read_json(previous_root / "completion.json", plan["diagnostic_completion_sha256"])
    if (not completion["completed"] or completion["manifest_sha256"] != plan["manifest_sha256"]
            or completion["plan_sha256"] != plan["diagnostic_plan_sha256"]
            or plan["decoder_config"] != {"preprocessing": previous["config"]["preprocessing"]}
            or plan["fas_root"] != previous["fas_root"]
            or plan["source_manifest_sha256"] != MANIFEST_SHA
            or plan["preprocessing_sha256"] != previous["preprocessing_sha256"]
            or plan["msu_reader_sha256"] != previous["msu_reader_sha256"]):
        raise ValueError("preview lineage or decoder contracts changed")
    read_json(Path(plan["manifest"]), plan["manifest_sha256"])
    videos, _ = load_manifest(Path(plan["manifest"]))
    expected = select_videos(videos)
    if object_digest([v.to_dict() for v in expected]) != object_digest([e["video"] for e in plan["selected"]]):
        raise ValueError("preview must retain its entire predetermined sample")
    source = read_json(Path(plan["source_manifest"]), MANIFEST_SHA)
    rows = {f"{r['dataset']}/{r['video_id']}": r for r in source["rows"]}
    for entry, video in zip(plan["selected"], expected):
        if (entry["samples"] != cache_samples(video)
                or any(entry[k] != value for k, value in source_media_entry(video, rows[video.key]).items())):
            raise ValueError("preview native-frame or archive binding changed")
    return plan


def read_native_frames(video: Video, path: Path, wanted: set[int], deadline: float,
                       replay=None) -> dict[int, np.ndarray]:
    if digest(path) != video.media_sha256:
        raise ValueError("preview staged bytes differ from accepted original")
    found, count = {}, 0

    def consume(index: int, rgb: np.ndarray) -> None:
        nonlocal count
        if time.monotonic() >= deadline:
            raise MediaFailure("preview_native_deadline")
        if (index != count or index >= len(video.timestamps) or rgb.dtype != np.uint8
                or rgb.ndim != 3 or rgb.shape[2] != 3):
            raise MediaFailure("preview_native_count_or_shape_mismatch")
        if hashlib.sha256(rgb.tobytes()).hexdigest() != video.frame_hashes[index]:
            raise MediaFailure("preview_native_RGB_mismatch_no_waiver")
        if index in wanted:
            found[index] = rgb.copy()
        count += 1

    if replay is not None:
        replay(path, consume, max(0.001, deadline - time.monotonic()))
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise MediaFailure("preview_video_open_failed")
            while True:
                success, bgr = capture.read()
                if not success:
                    break
                consume(count, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    if count != len(video.timestamps) or set(found) != wanted:
        raise MediaFailure("preview_native_population_mismatch")
    return found


def save_png(path: Path, rgb: np.ndarray) -> dict[str, str]:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError("preview PNG encoding failed")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())
    return {"file": str(Path("images") / path.name), "sha256": digest(path)}


def render_original(rgb: np.ndarray, quad: np.ndarray, legacy: dict[str, Any]) -> np.ndarray:
    result = rgb.copy()
    left, top, right, bottom = legacy["crop_xyxy_half_open"]
    corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float64)
    inverse = cv2.invertAffineTransform(np.asarray(legacy["alignment_matrix"]))
    old_quad = corners @ inverse[:, :2].T + inverse[:, 2]
    for polygon, color in ((old_quad, (245, 158, 11)), (quad, (6, 182, 212))):
        cv2.polylines(result, [np.rint(polygon).astype(np.int32)], True, color,
                      max(1, round(max(rgb.shape[:2]) / 400)), lineType=cv2.LINE_AA)
    scale = min(1., 640 / max(result.shape[:2]))
    if scale < 1:
        result = cv2.resize(result, (round(result.shape[1] * scale), round(result.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA)
    return result


def run_preview(path: Path, sha: str) -> None:
    plan = checked_inputs(path, sha)
    root = path.parent
    if (root / "started.json").exists():
        raise ValueError("preview already attempted; no automatic retry")
    write_json(root / "started.json", {"plan_sha256": sha})
    deadline = time.monotonic() + BUDGET["max_seconds"] - 20
    cv2.setNumThreads(1)
    fas_root = Path(plan["fas_root"])
    if (digest(fas_root / "src/fas/preprocessing.py") != plan["preprocessing_sha256"]
            or digest(fas_root / "scripts/inventory_msu.py") != plan["msu_reader_sha256"]):
        raise ValueError("preview source code bindings changed")
    sys.path.insert(0, str(fas_root / "src"))
    module = importlib.import_module("fas.preprocessing")
    if Path(module.__file__).resolve() != (fas_root / "src/fas/preprocessing.py").resolve():
        raise ValueError("preview imported an unbound legacy crop implementation")
    decoders, helper_sha = decoder_policy(plan["decoder_config"])
    native = None
    if any(e["video"]["dataset"] == "MSU-MFSD" for e in plan["selected"]):
        if helper_sha is None:
            raise ValueError("preview requires strict native MSU helper")
        native = FasNativeReplay(fas_root, helper_sha, Path(plan["source_manifest"]), MANIFEST_SHA)
    for name in ("images", "staging", "records"):
        (root / name).mkdir()
    with ExitStack() as stack:
        msu_context: dict[str, Any] = {}
        for index, entry in enumerate(plan["selected"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("preview shared wall cap")
            video = Video(**{**entry["video"], "timestamps": tuple(entry["video"]["timestamps"]),
                             "frame_hashes": tuple(entry["video"]["frame_hashes"])})
            staged = root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}"
            write_json(root / "records" / f"started-{index:02d}.json", {"index": index})
            rendered, failure = [], None
            try:
                stage_member(entry, fas_root, fas_root, staged, stack, msu_context, deadline,
                             allow_validation_attacks=True)
                replay = None
                if decoders[video.dataset] == "fas_strict_ffmpeg":
                    if native is None:
                        raise ValueError("strict replay unavailable")
                    replay = native.bind(replace(video, media_path=str(staged)),
                                         root / "native-replay" / f"original-{index:02d}",
                                         expected_receipt_sha256=entry["receipt_sha256"])
                frames = read_native_frames(video, staged, {s["native_order"] for s in entry["samples"]},
                                            min(deadline, time.monotonic() + 180), replay)
                with np.load(video.cache_path, allow_pickle=False) as cache:
                    old_frames = cache["frames"]
                    for sample in entry["samples"]:
                        rgb = frames[sample["native_order"]]
                        landmarks = np.asarray(sample["landmarks"])
                        legacy, legacy_meta = module.shared_context_crop(rgb, np.asarray(sample["box"]), landmarks)
                        legacy = cv2.resize(legacy, (256, 256), interpolation=cv2.INTER_CUBIC)
                        if not np.array_equal(legacy, old_frames[sample["cache_index"]]):
                            raise ValueError("preview legacy pixels differ from locked cache")
                        corrected, geometry = crop_celebahq(rgb, landmarks)
                        annotated = render_original(rgb, np.asarray(geometry["quad"]), legacy_meta)
                        prefix = f"v{index + 1:02d}-f{sample['clip_slot'] + 1:02d}"
                        images = {name: save_png(root / "images" / f"{prefix}-{name}.png", image)
                                  for name, image in (("original", annotated), ("legacy", legacy),
                                                      ("celebahq", corrected))}
                        rendered.append({"clip_slot": sample["clip_slot"],
                                         "native_timestamp": sample["native_timestamp"],
                                         "native_RGB_sha256": video.frame_hashes[sample["native_order"]],
                                         "native_width_height": [rgb.shape[1], rgb.shape[0]],
                                         "geometry": geometry, "images": images,
                                         "legacy_cache_pixels_exact": True})
                        if tree_bytes(root) > BUDGET["max_output_bytes"]:
                            raise OSError("preview output cap")
            except MediaFailure as error:
                failure = str(error)
            finally:
                staged.unlink(missing_ok=True)
            write_json(root / "records" / f"result-{index:02d}.json", {
                "index": index, "status": "technical_failure" if failure else "rendered",
                "failure": failure, "frames": rendered})
            print({"originals": index + 1, "selected": 12,
                   "status": "technical_failure" if failure else "rendered"}, flush=True)
    review = render_review(path)
    write_json(root / "completion.json", {
        "completed": True, "plan_sha256": sha, "human_review": "pending", "model_scoring_runs": 0,
        "html_sha256": review["html_sha256"], "counts": review["counts"],
        "records_sha256": {p.name: digest(p) for p in sorted((root / "records").glob("result-*.json"))},
    })


def render_review(path: Path) -> dict[str, Any]:
    root, plan = path.parent, read_json(path)
    require_private_output(root / "review.html")
    cards, counts = [], Counter()

    def embed(image: dict[str, str], label: str) -> str:
        asset = root / image["file"]
        if not asset.resolve().is_relative_to((root / "images").resolve()) or digest(asset) != image["sha256"]:
            raise ValueError("preview image binding mismatch")
        data = base64.b64encode(asset.read_bytes()).decode("ascii")
        return f'<figure><figcaption>{label}</figcaption><img alt="{label}" src="data:image/png;base64,{data}"></figure>'

    for index, entry in enumerate(plan["selected"]):
        receipt = root / "records" / f"result-{index:02d}.json"
        row = read_json(receipt) if receipt.exists() else {
            "status": "interrupted" if (root / "records" / f"started-{index:02d}.json").exists()
            else "unattempted", "failure": "No completed receipt", "frames": []}
        counts[row["status"]] += 1
        content = []
        for frame in row["frames"]:
            counts["frames"] += 1
            images = frame["images"]
            geometry = frame["geometry"]
            content.append(
                f'<section><h3>Clip frame {frame["clip_slot"] + 1}/16'
                f' &middot; native time {frame["native_timestamp"]:.3f}s</h3><div class="comparison">'
                + embed(images["original"], "Original + crop outlines (thumbnail)")
                + embed(images["legacy"], "A: legacy stretched crop, 256 x 256")
                + embed(images["celebahq"], "B: CelebA-HQ-style square crop, 256 x 256")
                + f'</div><p class="meta">Reflected padding: {geometry["padding_fraction"]:.2%}'
                f' &middot; square side: {geometry["side_source_pixels"]:.1f} source pixels'
                f' &middot; output eye distance: {geometry["eye_distance_output_pixels"]:.1f}px</p></section>')
        if row["failure"]:
            content.append(f'<p class="failure">{html.escape(row["failure"])}</p>')
        video = entry["video"]
        cards.append(f'<details open><summary>Preview {index + 1:02d} &middot; '
                     f'{html.escape(video["dataset"])} &middot; {html.escape(video["label"])}'
                     f' <span class="badge">{html.escape(row["status"])}</span></summary>'
                     + "".join(content) + "</details>")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Private DDPM crop review</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.5 system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:28px}header{background:#0f172a;color:#f8fafc;padding:28px;border-radius:16px;margin-bottom:24px}
h1{margin:8px 0;font-size:30px}.eyebrow{font-size:12px;letter-spacing:2px;color:#67e8f9}
.notice{border-left:4px solid #f59e0b;padding-left:14px}.meta{font-size:13px;color:#475569}
header .meta{color:#cbd5e1}.amber{color:#fbbf24}.cyan{color:#22d3ee}
details{background:white;border:1px solid #cbd5e1;border-radius:12px;margin:20px 0;overflow:hidden}
summary{padding:18px;cursor:pointer;font-weight:650;background:#e2e8f0}.badge{font-size:12px;border:1px solid #94a3b8;border-radius:20px;padding:3px 9px;margin-left:10px}
section{padding:18px;border-top:1px solid #e2e8f0}h3{font-size:14px;margin:0 0 14px}
.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}
figure{margin:0;min-width:0}figcaption{font-size:13px;margin-bottom:10px}
img{display:block;max-width:100%;height:auto;max-height:360px;object-fit:contain;margin:auto;background:#0f172a;border-radius:6px}
.failure{color:#991b1b;background:#fee2e2;padding:16px}.hash{overflow-wrap:anywhere}
@media(max-width:760px){main{padding:12px}.comparison{grid-template-columns:1fr}h1{font-size:24px}summary{font-size:14px}}
</style></head><body><main><header><div class="eyebrow">PRIVATE / PREPROCESSING REVIEW</div>
<h1>DDPM input geometry: before / after</h1>
<p class="notice"><strong>PAUSED FOR HUMAN REVIEW.</strong> No DDPM scoring, training, calibration or target access.
Approval is not automatic: review these crops in chat before any ablation run.</p>
<p>12 seeded random source-validation videos, 2 per domain/class. Three fixed frames per middle T16 clip.
Same native RGB pixels and cached smoothed landmarks for both crops; no new detector.</p>
<p><span class="amber">Amber outline: legacy context</span> &middot; <span class="cyan">Cyan outline: new square</span>.
The original panel is a thumbnail; both model crops are lossless 256 x 256 PNG.</p>
<p>New crop: eye/mouth landmark square geometry, reflected boundaries, bilinear 1024 then area 256.
No beautification, face restoration or learned super-resolution. Not bit-exact CelebA-HQ processing.</p>
<p>Inspect head/chin coverage, face scale, alignment, distortion, reflected edges and frame-to-frame stability.
Failures stay visible; no replacement examples were selected.</p>
"""
    page += (f'<p class="meta">{counts["rendered"]}/12 videos rendered; {counts["frames"]}/36 frames. '
             f'Pool: {plan["pool_prepared"]}/{plan["pool_originals"]} previously prepared originals. '
             f'Sampling seed: {SEED}.</p>'
             f'<p class="meta hash">Plan SHA-256: {digest(path)}</p></header>'
             + "".join(cards) + "</main></body></html>")
    encoded = page.encode("utf-8")
    if tree_bytes(root) + len(encoded) + 1024**2 > BUDGET["max_output_bytes"]:
        raise OSError("self-contained HTML exceeds preview output cap")
    with (root / "review.html").open("xb") as stream:
        stream.write(encoded)
    return {"html_sha256": digest(root / "review.html"), "counts": dict(counts)}


def summarize_preview(path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    root = path.parent
    plan = read_json(path)
    counts = Counter()
    for index, entry in enumerate(plan["selected"]):
        staged = root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}"
        staged.unlink(missing_ok=True)
        receipt = root / "records" / f"result-{index:02d}.json"
        if receipt.exists():
            row = read_json(receipt)
            if row["index"] != index:
                raise ValueError("preview receipt identity mismatch")
            counts[row["status"]] += 1
            counts["frames"] += len(row["frames"])
        else:
            counts["interrupted" if (root / "records" / f"started-{index:02d}.json").exists()
                   else "unattempted"] += 1
    completion = read_json(root / "completion.json") if (root / "completion.json").exists() else None
    if completion is not None:
        if (completion["plan_sha256"] != digest(path)
                or completion["html_sha256"] != digest(root / "review.html")
                or completion["counts"] != dict(counts) or len(completion["records_sha256"]) != 12):
            raise ValueError("preview completion binding mismatch")
        for name, expected in completion["records_sha256"].items():
            if digest(root / "records" / name) != expected:
                raise ValueError("preview receipt commitment mismatch")
    return {"completed": bool(supervision["completed"] and completion and completion["completed"]),
            "human_review": "pending", "automatic_continuation": False, "model_scoring_runs": 0,
            "training_runs": 0, "calibration_pixel_access": 0, "target_pixel_access": 0,
            "plan_sha256": digest(path), "html_sha256": completion["html_sha256"] if completion else None,
            "counts": dict(counts), "budget": BUDGET, "supervision": supervision,
            "output_bytes": tree_bytes(root), "sampling_replacements": 0}
