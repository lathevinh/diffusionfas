"""Bounded real-pixel crop cache and fixed common-cohort A/B inventory."""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .io import digest, object_digest, read_json, require_private_output, write_json
from .regional_study import inputs, transaction_for
from .spatial_crop_preview import (
    CANDIDATE, CropSupportError, load_crop_pair, require_bicubic_support,
)
from .spatial_diagnosis import completed_inputs, crop_candidates
from .spatial_dino import SEED, SETTINGS, patch_prior, prior_image
from .spatial_preparation import code_pins
from .spatial_study import ROLES, population

BUDGET = {"max_seconds": 3600, "max_rss_bytes": 4*1024**3, "max_output_bytes": 4*1024**3}
CROPS = ("legacy", "square")
ARMS = ("reference", "content", "anatomical")
POLICY = {
    "crop": "expanded_legacy_square_same_center_rotation_expand_shorter_axis",
    "source_interpolation": "bicubic_native_scale_then_bicubic_RGB224",
    "padding_shifting_shrinking_allowed": False,
    "common_eligibility": "original_spatial_usable_and_both_real_pixel_footprints_and_landmark_support",
    "selection_uses_scores": False, "source_roles_repartitioned": False,
    "shared_failures": "retained_original_denominators_for_both_crops_all_three_arms",
    "future_reporting": "both_common_cohort_conditional_and_original_denominator_system_metrics",
    "new_fits_proposed": 24, "new_calibrators_proposed": 24,
    "prior": "existing_soft_prior_with_crop_specific_pixel_center_landmarks",
    "historical_full_source_checkpoints_reused": False,
    "current_authority": "CPU_crop_preparation_only_no_training_or_calibration",
    "GPU_inference": False, "new_native_video_decode": False, "new_detector": False,
}


def footprint_check(transaction: dict, candidate: dict) -> None:
    width, height = transaction["selected_frame"]["width"], transaction["selected_frame"]["height"]
    geometry = transaction["geometry"]
    require_bicubic_support(np.asarray(geometry["alignment_matrix"]),
                            tuple(geometry["crop_xyxy_half_open"]), width, height)
    side = int(candidate["side_source_pixels"])
    require_bicubic_support(np.asarray(candidate["source_to_square_render_matrix"]),
                            (0, 0, side, side), width, height)


def inventory(contexts: list[dict], indexed: dict, entries: list[dict], services: dict) -> tuple[list, list]:
    by_key = {entry["key"]: entry for entry in entries}
    if len(by_key) != len(entries) or set(by_key) != set(indexed):
        raise ValueError("exact original common-cohort identity inventory required")
    jobs, roles = [], []
    for context in contexts:
        sources = {indexed[key]["dataset"] for key in context["fitting"]}
        if len(sources) != 3 or context["target"] in sources:
            raise ValueError("three training sources must exclude held-out target")
        pools = {role: [indexed[key] for key in context[role]
                        if by_key[key]["status"] == "usable"] for role in ROLES}
        services["study"].isolation([pools[role] for role in ROLES])
        strata = {}
        for role, rows in pools.items():
            domains = {context["target"]} if role == "evaluation" else sources
            if {row["dataset"] for row in rows} != domains:
                raise ValueError("common cohort lost a required source/target domain")
            counts = Counter((row["dataset"], row["binary_label"]) for row in rows)
            if any(counts[domain, label] < 1 for domain in domains for label in ("attack", "bona_fide")):
                raise ValueError("common cohort lost a required domain/class stratum")
            strata[role] = [{"dataset": domain, "label": label, "usable": counts[domain, label]}
                            for domain in sorted(domains) for label in ("attack", "bona_fide")]
        target_originals = [indexed[key] for key in context["evaluation"]]
        roles.append({
            "target": context["target"],
            "common_keys": {role: [key for key in context[role] if by_key[key]["status"] == "usable"]
                            for role in ROLES},
            "role_counts": {role: len(rows) for role, rows in pools.items()}, "strata": strata,
            "target_original_counts": dict(Counter(row["binary_label"] for row in target_originals)),
            "target_common_counts": dict(Counter(row["binary_label"] for row in pools["evaluation"])),
            "target_shared_failures": len(target_originals)-len(pools["evaluation"]),
        })
        for crop in CROPS:
            for arm in ARMS:
                jobs.append({
                    "target": context["target"], "crop": crop, "arm": arm, "seed": SEED,
                    "usable": {role: len(rows) for role, rows in pools.items()},
                    "maximum_optimizer_steps": ((len(pools["fitting"])+SETTINGS["batch"]-1)
                                                //SETTINGS["batch"])*SETTINGS["epochs"],
                })
    if len(jobs) != 24 or len({(job["target"], job["crop"], job["arm"]) for job in jobs}) != 24:
        raise ValueError("exact four-target two-crop three-arm inventory required")
    return jobs, roles


def encode_square(path: Path, pixels: np.ndarray) -> dict:
    if pixels.dtype != np.uint8 or pixels.shape != (224, 224, 3):
        raise ValueError("finite native-rendered uint8 RGB224 square required")
    okay, encoded = cv2.imencode(".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    if not okay:
        raise RuntimeError("square PNG encoding failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())
    with Image.open(path) as image:
        if image.mode != "RGB" or not np.array_equal(np.asarray(image), pixels):
            raise ValueError("saved square PNG roundtrip differs")
    return {"path": str(path.resolve()), "sha256": digest(path),
            "RGB_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "shape": [224, 224, 3]}


def run(completed_root: Path, diagnosis_root: Path, preview_root: Path, output: Path) -> dict:
    require_private_output(output/"preparation.json")
    if output.exists():
        raise FileExistsError("new immutable full-cohort crop preparation root required")
    started = time.monotonic()

    def guard() -> None:
        if time.monotonic()-started >= BUDGET["max_seconds"]-30:
            raise TimeoutError("full-cohort crop preparation reached closure reserve")

    old, _, _, origin = completed_inputs(completed_root)
    diagnosis = read_json(diagnosis_root/"summary.json")
    previous = read_json(diagnosis_root/"crop-records.json", diagnosis["crop_records_sha256"])
    preview = read_json(preview_root/"summary.json")
    if (diagnosis["completed"] is not True or preview["completed"] is not True
            or diagnosis["origin"]["plan.json"] != origin["plan.json"]
            or preview["origin_plan_sha256"] != origin["plan.json"]
            or preview["diagnosis_summary_sha256"] != digest(diagnosis_root/"summary.json")
            or preview["candidate"] != CANDIDATE or preview["frames"] != 16
            or digest(preview_root/"review.html") != preview["HTML_sha256"]
            or digest(preview_root/"records.json") != preview["records_sha256"]):
        raise ValueError("completed selected-geometry preview and matching diagnosis required")
    services, rows, _ = inputs(Path(old["fas_root"]))
    indexed = population(old, rows, services)
    audited = {record["key"]: record for record in previous}
    if len(audited) != 7505 or set(audited) != set(indexed):
        raise ValueError("exact audited original population required")
    files = {**old["files"], **{str(completed_root/name): sha for name, sha in origin.items()}}
    for root, names in ((diagnosis_root, ("summary.json", "crop-records.json")),
                         (preview_root, ("summary.json", "review.html", "records.json"))):
        for name in names:
            files[str(root/name)] = digest(root/name)
    output.mkdir(parents=True)
    write_json(output/"started.json", {"scope": POLICY, "origin": origin, "budget": BUDGET})
    entries, exclusions, encoded_bytes = [], Counter(), 0
    for slot, historical in enumerate(old["entries"]):
        guard()
        key = historical["key"]
        row, audit = indexed[key], audited[key]
        entry = {"key": key, "dataset": row["dataset"], "label": row["binary_label"],
                 "historical_status": historical["status"], "status": "failed",
                 "reason": historical["status"], "crops": None, "landmarks224": None}
        if historical["status"] == "usable":
            transaction = transaction_for(row)
            geometry = crop_candidates(transaction)
            candidate = geometry["candidates"][CANDIDATE]
            expected = audit["geometry"]["candidates"][CANDIDATE]
            for field in ("new_crop_eligible", "strict_pair_eligible", "source_quad", "landmarks224",
                          "source_to_RGB224_matrix", "side_source_pixels"):
                if candidate[field] != expected[field]:
                    raise ValueError(f"audited square geometry drift: {field}")
            if not candidate["strict_pair_eligible"]:
                entry["reason"] = "metadata_square_or_legacy_real_pixel_support_ineligible"
            else:
                try:
                    footprint_check(transaction, candidate)
                except CropSupportError as error:
                    entry["reason"] = "quantization_margin_support_ineligible"
                    entry["support_detail"] = str(error)
                else:
                    native, square, actual, bindings = load_crop_pair(row, transaction)
                    files.update(bindings)
                    landmarks = {"legacy": historical["landmarks224"],
                                 "square": candidate["landmarks224"]}
                    for crop, points in landmarks.items():
                        prior = patch_prior(prior_image(np.asarray(points))[None])
                        if prior.shape != (1, 256) or not np.isfinite(prior).all():
                            raise ValueError(f"crop-specific prior support invalid: {crop}")
                    asset = encode_square(output/"crops"/f"{slot:05d}.png", square)
                    files[asset["path"]] = asset["sha256"]
                    encoded_bytes += Path(asset["path"]).stat().st_size
                    entry.update({"status": "usable", "reason": None,
                                  "crops": {"legacy": row["crops"]["224"], "square": asset},
                                  "landmarks224": landmarks,
                                  "source_to_square_matrix": candidate["source_to_RGB224_matrix"],
                                  "source_to_square_render_matrix": candidate["source_to_square_render_matrix"],
                                  "square_side_source_pixels": candidate["side_source_pixels"],
                                  "legacy_exact_pixel_match": True,
                                  "minimum_native_square_margin_pixels": candidate["minimum_source_margin_pixels"],
                                  "interpolation_support_check": actual["support_check"],
                                  "source_frame_RGB_sha256": transaction["selected_frame"]["rgb_sha256"]})
                    del native, square, actual, prior
        if entry["status"] != "usable":
            exclusions[entry["reason"]] += 1
        entries.append(entry)
        if (slot+1) % 250 == 0:
            print("SPATIAL_CROP_PREPARED", slot+1, "of", len(old["entries"]),
                  "usable", sum(item["status"] == "usable" for item in entries), flush=True)
            gc.collect()
    jobs, roles = inventory(old["contexts"], indexed, entries, services)
    by_dataset = []
    for domain in sorted({entry["dataset"] for entry in entries}):
        for label in ("attack", "bona_fide"):
            selected = [entry for entry in entries if (entry["dataset"], entry["label"]) == (domain, label)]
            by_dataset.append({"dataset": domain, "label": label, "originals": len(selected),
                               "usable": sum(entry["status"] == "usable" for entry in selected),
                               "exclusions": dict(Counter(entry["reason"] for entry in selected
                                                          if entry["status"] != "usable"))})
    project = Path(__file__).resolve().parents[2]
    preparation_settings = {**SETTINGS,
                            "geometry": "paired_crop_specific_RGB224_pixel_center_landmarks"}
    preparation = {
        "study": "spatial_crop_preparation_v1", "completed": True, "policy": POLICY, "budget": BUDGET,
        "origin": origin, "fas_root": old["fas_root"], "settings": preparation_settings,
        "code_sha256": code_pins(project), "files": files, "entries": entries,
        "contexts": old["contexts"], "groups": old["groups"], "jobs": jobs, "roles": roles,
        "common_eligibility_sha256": object_digest([[e["key"], e["status"], e["reason"]] for e in entries]),
        "scientific_execution_authorized": False, "training_fits": 0, "calibrator_fits": 0,
        "native_video_decodes": 0, "GPU_inference": 0,
    }
    guard()
    path = output/"preparation.json"
    write_json(path, preparation)
    usable = sum(entry["status"] == "usable" for entry in entries)
    result = {
        "completed": True, "preparation_sha256": digest(path), "originals": len(entries),
        "common_usable": usable, "shared_failures": len(entries)-usable,
        "exclusions": dict(exclusions), "dataset_class_counts": by_dataset,
        "roles": [{key: value for key, value in role.items() if key != "common_keys"} for role in roles],
        "fit_inventory": jobs, "maximum_optimizer_steps": sum(job["maximum_optimizer_steps"] for job in jobs),
        "square_PNGs_written_and_roundtrip_checked": usable, "legacy_exact_matches": usable,
        "crop_specific_priors_checked": usable*2, "actual_native_pixel_support_pairs_checked": usable,
        "source_png_pixels_decoded": usable, "new_native_video_decodes": 0,
        "encoded_square_PNG_bytes": encoded_bytes, "bound_file_count": len(files),
        "policy": POLICY, "budget": BUDGET, "wall_seconds": time.monotonic()-started,
        "training_fits": 0, "calibrator_fits": 0, "GPU_inference": 0,
        "scientific_execution_authorized": False, "executable_training_runner_sealed": False,
    }
    write_json(output/"summary.json", result)
    return result
