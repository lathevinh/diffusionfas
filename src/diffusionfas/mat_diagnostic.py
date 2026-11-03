"""Fixed, approved MSU development diagnostic after native MAT visual acceptance."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import hashlib
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import torch

from .crop_ablation import checked_inputs as checked_ablation, summarize as summarize_ablation
from .crop_preview import read_native_frames
from .data import addressed_seed
from .fas_import import MANIFEST_SHA
from .gpu_probe import tensor_digest
from .io import digest, read_json, require_private_output, write_json
from .mat_input import POLICY, anatomical_masks, crop_native512, masked_error
from .mat_preview import (
    acquired_weights, checked_inputs as checked_preview, configure_mat_gpu, load_mat,
    reconstruct, smoke_test, summarize as summarize_preview,
)
from .metrics import Prediction, auroc_summary, paired_auroc_bootstrap
from .native_replay import FasNativeReplay, decoder_policy
from .preprocessing import MediaFailure
from .source_preparation import item_video
from .source_probe import stage_member
from .supervision import tree_bytes

BUDGET = {"max_seconds": 1800, "max_rss_bytes": 8 * 1024**3,
          "max_gpu_bytes": 8 * 1024**3, "max_output_bytes": 1024**3}
AUTHORITY = {"source_validation_MSU_only": True, "frozen_model_only": True,
             "approved_third_party_mirror_research_only": True, "AUROC": True,
             "training": False, "threshold_fitting": False, "fitting_pixels": False,
             "calibration_pixels": False, "target_pixels": False,
             "new_download": False, "automatic_retry": False}
SCORING = {"frames": 16, "window": "existing_lower_middle_T16", "draws_per_mask": 1,
           "mask_regions": POLICY["mask_regions"], "loss": "normalized_RGB_hole_only_MSE",
           "primary": "equal_mean_over_regions_then_frames", "higher_score_is_attack": True,
           "secondary": "four_predeclared_region_means_no_selection_or_sign_flip",
           "seed": 20261103, "seed_namespace": "MAT-preview"}
PREVIEW_SHA = "39c1df3b2f7dfe3c2ef26dcd7562b6dc9e6400637a648ddca46fa1b904c7a87f"
BASELINE_SHA = "d27ad239c51dd9078e4f2e819ee35e329796e1ab52ce1161648282a42d87319a"
BOOTSTRAP = {"repetitions": 2000, "seed": addressed_seed(20261103, "MSU-MFSD", "crop-ablation-bootstrap")}
SCOPE = "MSU_frozen_MAT_anatomical_development_diagnostic"


def accepted_inputs(preview_root: Path, baseline_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preview = checked_preview(preview_root / "plan.json", PREVIEW_SHA)
    review = summarize_preview(preview_root / "plan.json", read_json(preview_root / "supervision.json"))
    approval = read_json(preview_root / "visual-approval.json")
    if (not review["completed"] or review["completion"]["review"]["rendered_frames"] != 18
            or review["completion"]["review"]["reconstructions"] != 72
            or review["completion"]["review"]["technical_failure_originals"] != 0
            or review["smoke"]["same_seed_max_absolute_error"] != 0
            or review["smoke"]["hidden_ground_truth_change_max_absolute_error"] != 0
            or approval["crop_and_masks_accepted"] is not True
            or approval["preview_plan_sha256"] != PREVIEW_SHA
            or approval["html_sha256"] != digest(preview_root / "review.html")
            or approval["budget"] != BUDGET or approval["authority"] != AUTHORITY
            or approval["scoring"] != SCORING or approval["bootstrap"] != BOOTSTRAP):
        raise PermissionError("completed MAT preview and explicit fixed diagnostic approval required")
    baseline = checked_ablation(baseline_root / "plan.json", BASELINE_SHA)
    report = summarize_ablation(baseline_root / "plan.json", read_json(baseline_root / "supervision.json"))
    if (not report["completed"] or report["result"]["originals"] != 22
            or report["result"]["model_scoring_profiles"] != 44
            or baseline["preview_root"] != preview["preview_root"]):
        raise ValueError("completed matched 22-original DDPM baseline required")
    return preview, baseline


def build_plan(preview_root: Path, baseline_root: Path, output: Path) -> Path:
    require_private_output(output / "plan.json")
    if output.exists():
        raise ValueError("new MAT diagnostic root required; no retries")
    preview, baseline = accepted_inputs(preview_root, baseline_root)
    weights_root = Path(preview["reuse_acquisition_root"])
    acquired_weights(weights_root)
    project = Path(__file__).resolve().parents[2]
    plan = {"scope": SCOPE, "budget": BUDGET, "authority": AUTHORITY, "scoring": SCORING,
            "bootstrap": BOOTSTRAP, "policy": POLICY, "entries": baseline["entries"],
            "preview_root": str(preview_root.resolve()), "baseline_root": str(baseline_root.resolve()),
            "acceptance_sha256": digest(preview_root / "visual-approval.json"),
            "baseline_result_sha256": digest(baseline_root / "result.json"),
            "weights_root": str(weights_root), "upstream_root": preview["upstream_root"],
            "fas_root": baseline["fas_root"], "source_manifest": baseline["source_manifest"],
            "msu_reader_sha256": baseline["msu_reader_sha256"],
            "decoder_config": preview["decoder_config"],
            "implementation_sha256": {
                str(p.relative_to(project)): digest(p) for directory in ("src", "scripts")
                for p in sorted((project / directory).rglob("*.py"))}}
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def checked_inputs(path: Path, sha: str) -> dict[str, Any]:
    require_private_output(path)
    plan = read_json(path, sha)
    if (plan["scope"] != SCOPE or plan["authority"] != AUTHORITY or plan["budget"] != BUDGET
            or plan["scoring"] != SCORING or plan["bootstrap"] != BOOTSTRAP or plan["policy"] != POLICY):
        raise PermissionError("fixed explicitly approved MAT diagnostic and resource caps required")
    project = Path(__file__).resolve().parents[2]
    expected = {str(p.relative_to(project)): digest(p) for directory in ("src", "scripts")
                for p in sorted((project / directory).rglob("*.py"))}
    if expected != plan["implementation_sha256"]:
        raise ValueError("MAT diagnostic implementation drift")
    preview_root, baseline_root = Path(plan["preview_root"]), Path(plan["baseline_root"])
    read_json(preview_root / "visual-approval.json", plan["acceptance_sha256"])
    read_json(baseline_root / "result.json", plan["baseline_result_sha256"])
    preview, baseline = accepted_inputs(preview_root, baseline_root)
    if (plan["entries"] != baseline["entries"] or plan["weights_root"] != preview["reuse_acquisition_root"]
            or plan["upstream_root"] != preview["upstream_root"]
            or plan["decoder_config"] != preview["decoder_config"]
            or any(plan[k] != baseline[k] for k in ("fas_root", "source_manifest", "msu_reader_sha256"))):
        raise ValueError("MAT population, full T16, native decoder or model binding drift")
    acquired_weights(Path(plan["weights_root"]))
    return plan


def aggregate_profile(profile: np.ndarray) -> dict[str, float]:
    profile = np.asarray(profile, dtype=np.float64)
    if profile.shape != (16, 4) or not np.isfinite(profile).all() or np.any(profile < 0):
        raise ValueError("finite nonnegative 16-frame by four-region MSE profile required")
    return {"primary": float(profile.mean(axis=1).mean()),
            **{name: float(profile[:, i].mean()) for i, name in enumerate(POLICY["mask_regions"])}}


def score_frames(model, video, samples: list[dict[str, Any]], frames: dict[int, np.ndarray],
                 device: torch.device) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if len(samples) != 16 or [s["clip_slot"] for s in samples] != list(range(16)):
        raise ValueError("exact committed middle T16 sample order required")
    records, crops, arrays = [], [], {}
    for sample in samples:
        rgb, geometry = crop_native512(frames[sample["native_order"]], np.asarray(sample["landmarks"]))
        crops.append(rgb)
        regions = {}
        for name, known in anatomical_masks(np.asarray(geometry["output_landmarks"])).items():
            seed = addressed_seed(SCORING["seed"], video.key, sample["clip_slot"], name, SCORING["seed_namespace"])
            prediction = reconstruct(model, rgb, known, seed, device)
            mse, _ = masked_error(rgb, prediction, known)
            hole = known == 0
            arrays[f"f{sample['clip_slot']:02d}-{name}"] = prediction[hole].copy()
            regions[name] = {"seed": seed, "masked_mse": mse, "hole_pixels": int(hole.sum()),
                             "known_mask_sha256": hashlib.sha256(known.tobytes()).hexdigest()}
        records.append({"clip_slot": sample["clip_slot"], "geometry": geometry, "masks": regions,
                        "crop_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "native_RGB_sha256": video.frame_hashes[sample["native_order"]]})
    arrays["crops"] = np.stack(crops)
    return records, arrays


def predictions(plan: dict[str, Any], root: Path) -> dict[str, list[Prediction]]:
    arms: dict[str, list[Prediction]] = {name: [] for name in ("DDPM_real_pixels", "MAT_primary", *POLICY["mask_regions"])}
    for index, entry in enumerate(plan["entries"]):
        video = item_video(entry)
        receipt = read_json(root / "records" / f"result-{index:02d}.json")
        if receipt["key"] != video.key:
            raise ValueError("MAT score receipt identity drift")
        scores = None
        if receipt["failure"] is None:
            profile = [[frame["masks"][name]["masked_mse"] for name in POLICY["mask_regions"]]
                       for frame in receipt["frames"]]
            scores = aggregate_profile(np.asarray(profile))
        baseline = read_json(Path(plan["baseline_root"]) / "records" / f"score-{index:02d}.json")["arms"]["real_pixels"]
        row = Prediction(**{k: v for k, v in baseline.items() if k != "profile"})
        if (row.key, row.dataset, row.group, row.partition, row.label) != (
                video.key, video.dataset, video.group, video.partition, video.label):
            raise ValueError("paired MAT/DDPM prediction identity drift")
        arms["DDPM_real_pixels"].append(row)
        for name in ("MAT_primary", *POLICY["mask_regions"]):
            score = scores["primary" if name == "MAT_primary" else name] if scores is not None else None
            arms[name].append(Prediction(video.key, video.dataset, video.group, video.partition,
                                         video.label, score, receipt["failure"], video.attack_family))
    return arms


def run_diagnostic(path: Path, sha: str) -> None:
    plan = checked_inputs(path, sha)
    root = path.parent
    if (root / "started.json").exists():
        raise ValueError("MAT diagnostic already attempted; no automatic retry")
    write_json(root / "started.json", {"plan_sha256": sha})
    deadline = time.monotonic() + BUDGET["max_seconds"] - 20
    for folder in ("records", "staging", "pixels"):
        (root / folder).mkdir()
    write_json(root / "device.json", configure_mat_gpu(plan))
    model = load_mat(plan, Path(plan["weights_root"]) / "weights/model.safetensors")
    state_sha = tensor_digest(model.state_dict())
    write_json(root / "smoke.json", smoke_test(model))
    print({"phase": "synthetic_smoke", "completed": True}, flush=True)
    cv2.setNumThreads(1)
    fas_root = Path(plan["fas_root"])
    if digest(fas_root / "scripts/inventory_msu.py") != plan["msu_reader_sha256"]:
        raise ValueError("MSU reader drift")
    decoders, helper_sha = decoder_policy(plan["decoder_config"])
    if decoders["MSU-MFSD"] != "fas_strict_ffmpeg" or helper_sha is None:
        raise ValueError("strict MSU native replay required")
    native = FasNativeReplay(fas_root, helper_sha, Path(plan["source_manifest"]), MANIFEST_SHA)
    torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        context: dict[str, Any] = {}
        for index, entry in enumerate(plan["entries"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("MAT shared wall cap")
            video = item_video(entry)
            staged = root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}"
            rendered, failure, pixels_sha = [], None, None
            try:
                stage_member(entry, fas_root, fas_root, staged, stack, context, deadline,
                             allow_validation_attacks=True)
                replay = native.bind(replace(video, media_path=str(staged)),
                                     root / "native-replay" / f"original-{index:02d}",
                                     expected_receipt_sha256=entry["receipt_sha256"])
                frames = read_native_frames(video, staged, {s["native_order"] for s in entry["samples"]},
                                            min(deadline, time.monotonic() + 180), replay)
                rendered, arrays = score_frames(model, video, entry["samples"], frames, torch.device("cuda"))
                pixels = root / "pixels" / f"{index:02d}.npz"
                with pixels.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                pixels_sha = digest(pixels)
                del arrays, frames
            except MediaFailure as error:
                failure = str(error)
            finally:
                staged.unlink(missing_ok=True)
            write_json(root / "records" / f"result-{index:02d}.json",
                       {"key": video.key, "failure": failure, "frames": rendered, "pixels_sha256": pixels_sha})
            if tree_bytes(root) > BUDGET["max_output_bytes"]:
                raise OSError("MAT diagnostic output cap")
            print({"phase": "MAT_scoring", "originals": index + 1, "total": 22, "failure": failure}, flush=True)
    if tensor_digest(model.state_dict()) != state_sha:
        raise ValueError("MAT frozen model state changed")
    arms = predictions(plan, root)
    primary = {k: arms[k] for k in ("DDPM_real_pixels", "MAT_primary")}
    summaries = {name: auroc_summary(rows) for name, rows in arms.items()}
    a, b = [summaries[name]["auroc_usable"] for name in primary]
    write_json(root / "result.json", {
        "scope": SCOPE, "plan_sha256": sha, "summaries": summaries,
        "MAT_primary_minus_DDPM": b - a if a is not None and b is not None else None,
        "paired_subject_bootstrap": paired_auroc_bootstrap(primary, **BOOTSTRAP),
        "small_group_count_warning": True, "originals": 22,
        "frames": sum(len(read_json(p)["frames"]) for p in sorted((root / "records").glob("result-*.json"))),
        "reconstructions": 64 * sum(row.score is not None for row in arms["MAT_primary"]),
        "training_runs": 0, "threshold_fitting_runs": 0, "new_download_bytes": 0,
        "fitting_pixel_access": 0, "calibration_pixel_access": 0, "target_pixel_access": 0,
        "frozen_model_sha256_before_and_after": state_sha,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "records_sha256": {p.name: digest(p) for p in sorted((root / "records").glob("*.json"))}})


def summarize(path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    root, plan = path.parent, read_json(path)
    for index, entry in enumerate(plan["entries"]):
        (root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}").unlink(missing_ok=True)
    result = read_json(root / "result.json") if (root / "result.json").exists() else None
    if result is not None:
        if result["plan_sha256"] != digest(path):
            raise ValueError("MAT diagnostic result commitment drift")
        for name, expected in result["records_sha256"].items():
            row = read_json(root / "records" / name, expected)
            if row["failure"] is None and digest(root / "pixels" / f"{int(name[7:9]):02d}.npz") != row["pixels_sha256"]:
                raise ValueError("MAT unquantized pixel evidence drift")
    return {"completed": bool(supervision["completed"] and result is not None),
            "plan_sha256": digest(path), "authority": AUTHORITY, "scoring": SCORING,
            "policy": POLICY, "budget": BUDGET, "supervision": supervision,
            "result": {k: v for k, v in result.items() if k != "records_sha256"} if result is not None else None,
            "smoke": read_json(root / "smoke.json") if (root / "smoke.json").exists() else None,
            "development_subset_not_independent_test": True, "visual_review": "accepted"}
