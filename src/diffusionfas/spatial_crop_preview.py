"""Private real-pixel preview of sixteen score-independent DINO crop pairs."""

from __future__ import annotations

import base64
from collections import Counter
import hashlib
import html
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .io import digest, object_digest, read_json, require_private_output, write_json
from .regional_study import inputs, transaction_for
from .spatial_diagnosis import PLAN_SHA, completed_inputs, crop_candidates
from .spatial_study import population

BUDGET = {"max_seconds": 1200, "max_rss_bytes": 2*1024**3, "max_output_bytes": 256*1024**2}
SEED = 20261106
CANDIDATE = "expanded_legacy_square"


def preview_keys(plan: dict, records: list[dict]) -> list[str]:
    indexed = {r["key"]: r for r in records}
    if len(indexed) != len(records):
        raise ValueError("unique crop-audit identities required")
    selected = []
    for context in plan["contexts"]:
        for label in ("bona_fide", "attack"):
            candidates = [key for key in context["evaluation"]
                if indexed[key]["label"] == label and indexed[key]["geometry"] is not None
                and indexed[key]["geometry"]["candidates"][CANDIDATE]["strict_pair_eligible"]]
            candidates.sort(key=lambda key: object_digest({"seed": SEED, "purpose": "crop_preview",
                                                          "target": context["target"], "key": key}))
            if len(candidates) < 2:
                raise ValueError("two eligible target originals per class required")
            selected.extend(candidates[:2])
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("exact sixteen unique target middle frames required")
    return selected


def require_bicubic_support(matrix: np.ndarray, bounds: tuple[int, int, int, int],
                            width: int, height: int) -> None:
    left, top, right, bottom = bounds
    corners = np.asarray([[left, top], [right-1, top], [right-1, bottom-1], [left, bottom-1]], dtype=float)
    inverse = cv2.invertAffineTransform(matrix)
    source = corners@inverse[:, :2].T+inverse[:, 2]
    # Affine extrema occur at corners. Include cubic taps and OpenCV's
    # 1/32-pixel interpolation quantization without relying on mask sums.
    minimum = source.min(axis=0)-2-1/32
    maximum = source.max(axis=0)+2+1/32
    if np.any(minimum < 0) or np.any(maximum > [width-1, height-1]):
        raise ValueError("bicubic interpolation footprint outside real source pixels")


def render_pair(rgb: np.ndarray, transaction: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    frame = transaction["selected_frame"]
    if (rgb.dtype != np.uint8 or rgb.shape != (frame["height"], frame["width"], 3)
            or hashlib.sha256(rgb.tobytes()).hexdigest() != frame["rgb_sha256"]):
        raise ValueError("exact historical native selected RGB frame required")
    audit = crop_candidates(transaction)
    candidate = audit["candidates"][CANDIDATE]
    if not candidate["strict_pair_eligible"]:
        raise ValueError("selected crop pair lacks conservative real-pixel support")
    geometry = transaction["geometry"]
    matrix = np.asarray(geometry["alignment_matrix"])
    left, top, right, bottom = geometry["crop_xyxy_half_open"]
    height, width = rgb.shape[:2]
    require_bicubic_support(matrix, (left, top, right, bottom), width, height)
    aligned = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    old = cv2.resize(aligned[top:bottom, left:right], (224, 224), interpolation=cv2.INTER_CUBIC)
    side = int(candidate["side_source_pixels"])
    render = np.asarray(candidate["source_to_square_render_matrix"])
    require_bicubic_support(render, (0, 0, side, side), width, height)
    square = cv2.warpAffine(rgb, render, (side, side), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    new = cv2.resize(square, (224, 224), interpolation=cv2.INTER_CUBIC)
    return old, new, {
        **audit, "actual_legacy_interpolation_support_verified": True,
        "actual_square_interpolation_support_verified": True, "padding_fraction": 0.,
        "interpolation": "bicubic_native_scale_render_then_bicubic_RGB224_resize",
        "support_check": "affine_extrema_plus_two_pixel_radius_and_one_over_32_quantization_margin",
    }


def png(image: np.ndarray) -> str:
    okay, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not okay:
        raise RuntimeError("preview PNG encoding failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def run(completed_root: Path, diagnosis_root: Path, output: Path) -> dict:
    require_private_output(output/"review.html")
    if output.exists():
        raise FileExistsError("new immutable preview output root required")
    started = time.monotonic()

    def guard() -> None:
        if time.monotonic()-started >= BUDGET["max_seconds"]-30:
            raise TimeoutError("crop preview closure reserve reached")

    summary = read_json(diagnosis_root/"summary.json")
    if summary["completed"] is not True or summary["origin"]["plan.json"] != PLAN_SHA:
        raise ValueError("completed fixed-origin metadata diagnosis required")
    records = read_json(diagnosis_root/"crop-records.json", summary["crop_records_sha256"])
    plan, _, _, _ = completed_inputs(completed_root)
    services, rows, _ = inputs(Path(plan["fas_root"]))
    indexed = population(plan, rows, services)
    selected = preview_keys(plan, records)
    panels, receipts, bindings = [], [], {}
    for slot, key in enumerate(selected):
        guard()
        row = indexed[key]
        transaction = transaction_for(row)
        asset = transaction["selected_RGB_asset"]
        anchor = Path(row["record_path"]).parent.parent.resolve()
        relative = Path(asset["path"])
        raw = anchor/relative
        if relative.is_absolute() or not raw.resolve().is_relative_to(anchor):
            raise ValueError("native PNG asset escaped original transaction root")
        if (digest(raw) != asset["sha256"]
                or asset["RGB_sha256"] != transaction["selected_frame"]["rgb_sha256"]
                or asset["shape"] != [transaction["selected_frame"]["height"],
                                      transaction["selected_frame"]["width"], 3]):
            raise ValueError("native selected PNG commitment differs")
        with Image.open(raw) as image:
            if image.mode != "RGB":
                raise ValueError("native selected PNG must already be RGB")
            rgb = np.asarray(image)
        old, new, geometry = render_pair(rgb, transaction)
        crop = row["crops"]["224"]
        if digest(Path(crop["path"])) != crop["sha256"]:
            raise ValueError("historical RGB224 PNG binding changed")
        with Image.open(crop["path"]) as image:
            if image.mode != "RGB" or not np.array_equal(np.asarray(image), old):
                raise ValueError("regenerated legacy crop differs from pinned historical pixels")
        for path, sha in ((raw, asset["sha256"]), (Path(crop["path"]), crop["sha256"]),
                          (Path(row["record_path"]), row["record_sha256"])):
            bindings[str(path)] = sha
        candidate = geometry["candidates"][CANDIDATE]
        preview = rgb.copy()
        quad = np.rint(candidate["source_quad"]).astype(np.int32)
        cv2.polylines(preview, [quad], True, (0, 200, 0), thickness=3)
        cv2.polylines(preview, [np.rint(np.asarray(transaction["landmarks"])).astype(np.int32)],
                      False, (255, 80, 0), thickness=2)
        scale = min(512/rgb.shape[1], 512/rgb.shape[0], 1)
        preview = cv2.resize(preview, (max(1, round(rgb.shape[1]*scale)),
                                       max(1, round(rgb.shape[0]*scale))))
        marked = new.copy()
        for x, y in candidate["landmarks224"]:
            cv2.circle(marked, (round(x), round(y)), 2, (255, 80, 0), -1)
        title = html.escape(f"{slot:02d} {row['dataset']} {row['binary_label']}")
        panels.append(
            f"<section><h2>{title}</h2><div><figure><img src='data:image/png;base64,{png(preview)}'>"
            "<figcaption>Native frame: green square, orange landmarks</figcaption></figure>"
            f"<figure><img src='data:image/png;base64,{png(old)}'><figcaption>Historical RGB224</figcaption></figure>"
            f"<figure><img src='data:image/png;base64,{png(marked)}'>"
            "<figcaption>Expanded square RGB224 + landmarks</figcaption></figure></div></section>"
        )
        receipts.append({"key": key, "dataset": row["dataset"], "label": row["binary_label"],
                         "legacy_pixels_match": True, "geometry": geometry,
                         "new_RGB_sha256": hashlib.sha256(new.tobytes()).hexdigest()})
        del rgb, preview, old, new, marked
    output.mkdir(parents=True)
    page = (
        "<!doctype html><html><meta charset='utf-8'><title>DINO crop preview</title>"
        "<style>body{font-family:sans-serif}section{border-bottom:1px solid #aaa}"
        "section div{display:flex;flex-wrap:wrap}figure{margin:12px}img{max-width:512px}"
        "</style><h1>Private DINO crop preview: no training or scoring</h1>"
        "<p>16 fixed middle frames, selected without scores. Expanded square keeps the old center "
        "and rotation; adds context on the shorter axis. No padding, shifting or shrinking. "
        "Orange dots are landmark overlays, not model attention. All legacy pixels match.</p>"
        +"".join(panels)+"</html>"
    )
    with (output/"review.html").open("x") as stream:
        stream.write(page)
    write_json(output/"records.json", receipts)
    result = {
        "completed": True, "frames": 16, "embedded_PNGs": 48,
        "selection_seed": SEED, "selection_uses_scores": False, "candidate": CANDIDATE,
        "counts": dict(Counter(f"{r['dataset']}:{r['label']}" for r in receipts)),
        "legacy_exact_matches": 16, "real_pixel_support_pairs_verified": 16,
        "padding_fraction": 0., "budget": BUDGET, "bound_files": bindings,
        "origin_plan_sha256": PLAN_SHA, "diagnosis_summary_sha256": digest(diagnosis_root/"summary.json"),
        "records_sha256": digest(output/"records.json"), "HTML_sha256": digest(output/"review.html"),
        "new_native_video_decodes": 0, "new_detectors": 0, "GPU_inference": 0,
        "training": 0, "calibration": 0, "human_review_required": True,
        "wall_seconds": time.monotonic()-started,
    }
    guard()
    write_json(output/"summary.json", result)
    return result
