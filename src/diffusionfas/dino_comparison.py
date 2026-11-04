"""Approved matched-video, not matched-crop, frozen DINO/MAT diagnostic."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import gc
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

from .crop_preview import read_native_frames
from .fas_import import MANIFEST_SHA
from .gpu_probe import tensor_digest
from .io import digest, read_json, require_private_output, write_json
from .mat_diagnostic import predictions as mat_predictions, summarize as summarize_mat
from .mat_preview import configure_mat_gpu
from .metrics import Prediction, auroc_summary, paired_auroc_bootstrap
from .native_replay import FasNativeReplay, decoder_policy
from .preprocessing import MediaFailure
from .source_preparation import item_video
from .source_probe import stage_member
from .supervision import tree_bytes

BUDGET = {"max_seconds": 1800, "max_rss_bytes": 8 * 1024**3,
          "max_gpu_bytes": 8 * 1024**3, "max_output_bytes": 1024**3}
SEEDS = (20260917, 20260923, 20261001)
BRANCHES = ("dino_reg", "dino_plain")
MAT_SHA = "38a5621ccd6b80347f55dccd1d8b340a15703de5be95f8c183aed73b2f7910f8"
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
PARTITIONS_SHA = "2700baf9d757cc8c773247f104cd2a85c4ee9c288e461bb929cdbff9333c2996"
PUBLIC_SHA = "5dea308b3170310dc14017166e4af196fdcdfb1d95a35be41f56e1f7b12e32c6"
HEAD_SHAS = (
    "71e994bb46f22945a4287fab1a3cb5bf28e14ce5e4fee583e0c042f9668eaaca",
    "52108aa19fa12b763956885f66a39316d267423a22cd305ad2c0a38b38c92b63",
    "fe3e26287fe477854f9d65e0700022c769d46d251d2027a2b898d702f16bdde7",
)
WEIGHT_SHAS = (
    "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
    "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
)
AUTHORITY = {"source_validation_MSU_22_only": True, "all_three_existing_seeds": True,
             "MSU_excluded_from_all_head_fit_validation_calibration": True,
             "legacy_DINO_rectangle_to_square_resize_explicitly_accepted": True,
             "new_training": False, "threshold_selection": False, "fusion_fitting": False,
             "new_download": False, "new_calibration_or_target_pixels": False, "automatic_retry": False}
POLICY = {"name": "historical_DINO_native518_strict_real_support_v1", "frames": 16,
          "source": "native_RGB_not_upscaled_cache", "crop": "original_shared_context_crop",
          "context_scale": 1.30, "resize": "original_bicubic_rectangle_to_518_square",
          "aspect_ratio_distortion_possible": True, "same_crop_as_MAT": False,
          "padding_allowed": False, "source_cubic_4x4_footprint_required": True,
          "mean": [.485, .456, .406], "std": [.229, .224, .225],
          "pooling": "normalized_CLS_concat_mean_normalized_patch_excluding_registers",
          "video_score": "mean_frame_calibrated_attack_probability",
          "two_dino": "fixed_50_50_frame_probabilities_then_video_mean",
          "seed_selection": False, "cross_seed_model_ensemble": False,
          "complementarity": "3x3_attack_bonafide_pair_ordering_with_explicit_ties",
          "operating_point_rescue_or_FA_claim": False}


def fas_modules(root: Path) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    modules = {}
    for name in ("preprocessing", "frozen_features", "source_fitting"):
        module = importlib.import_module(f"fas.{name}")
        if Path(module.__file__).resolve() != (root / f"src/fas/{name}.py").resolve():
            raise ValueError("unbound original FAS module")
        modules[name] = module
    return modules


def validate_models(model: dict[str, Any], seed: int) -> None:
    for name in BRANCHES:
        head, calibration = model["heads"][name], model["calibrators"][name]
        coefficients = np.asarray(head["coefficients"])
        if (head["seed"] != seed or coefficients.shape != (1537,) or not np.isfinite(coefficients).all()
                or not np.isfinite([calibration["slope"], calibration["intercept"]]).all()
                or calibration["slope"] <= 0):
            raise ValueError("exact finite frozen DINO head and monotone calibrator required")


def validate_role_isolation(context: dict[str, Any], rows: dict[str, Any], queried_hashes: set[str]) -> None:
    for role in ("fitting", "validation", "calibration"):
        for key in context[role]:
            row = rows[key]
            if row["dataset"] == "MSU-MFSD":
                raise PermissionError("queried dataset entered a DINO fitting role")
            if row["detector_status"] == "success":
                frame = row["selected_frame"]
                if not isinstance(frame, dict) or not frame.get("rgb_sha256"):
                    raise ValueError("successful source input requires selected-frame evidence")
                if frame["rgb_sha256"] in queried_hashes:
                    raise PermissionError("queried exact frame entered a DINO fitting role")


def input_bindings(mat_root: Path, fas_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    mat = read_json(mat_root / "plan.json", MAT_SHA)
    project = Path(__file__).resolve().parents[2]
    for name, sha in mat["implementation_sha256"].items():
        if digest(project / name) != sha:
            raise ValueError("completed MAT implementation drift")
    report = summarize_mat(mat_root / "plan.json", read_json(mat_root / "supervision.json"))
    if (not report["completed"] or report["result"]["originals"] != 22
            or report["result"]["reconstructions"] != 1408
            or report["result"]["summaries"]["MAT_primary"]["coverage"]["attack"]["usable"] != 16
            or report["result"]["summaries"]["MAT_primary"]["coverage"]["bona_fide"]["usable"] != 6
            or mat["fas_root"] != str(fas_root.resolve())
            or read_json(mat_root / "postflight.json")["completed"] is not True):
        raise ValueError("completed verified fixed MAT cohort required")
    private = fas_root.parent / "FAS-private/artifacts/phase1"
    comparison = private / "paper_comparison323_v1"
    partition_path = private / "paper_completion319_v1/partitions.json"
    public = fas_root / "results/phase1/paper324-comparison-v1.json"
    closure = read_json(public, PUBLIC_SHA)
    bindings = {str(public): PUBLIC_SHA, str(partition_path): PARTITIONS_SHA}
    for name in ("execution_plan.json", "worker_completion.json", "verification.json", "terminal.json"):
        path = comparison / name
        sha = closure["source_sha256"][name]
        read_json(path, sha)
        bindings[str(path)] = sha
    original_plan = read_json(comparison / "execution_plan.json")
    for name in ("src/fas/frozen_features.py", "src/fas/source_fitting.py"):
        if digest(fas_root / name) != original_plan["implementation_sha256"][name]:
            raise ValueError("published DINO feature or head inference implementation drift")
    baseline_root = Path(mat["baseline_root"])
    baseline = read_json(baseline_root / "plan.json")
    if digest(fas_root / "src/fas/preprocessing.py") != baseline["preprocessing_sha256"]:
        raise ValueError("historical DINO/shared-context preprocessing drift")
    baseline_result = read_json(baseline_root / "result.json", mat["baseline_result_sha256"])
    for name, sha in baseline_result["records_sha256"].items():
        read_json(baseline_root / "records" / name, sha)
    if (read_json(comparison / "terminal.json")["status"] != "complete"
            or read_json(comparison / "verification.json")["all12_contexts_reconstructed"] is not True):
        raise ValueError("completed independently reconstructed original DINO experiment required")
    partitions = read_json(partition_path, PARTITIONS_SHA)
    rows = {f"{r['dataset']}/{r['video_id']}": r for r in read_json(Path(mat["source_manifest"]), MANIFEST_SHA)["rows"]}
    queried_hashes = {h for entry in mat["entries"] for h in entry["video"]["frame_hashes"]}
    for seed, sha in zip(SEEDS, HEAD_SHAS, strict=True):
        contexts = [c for c in partitions["contexts"] if c["target"] == "MSU-MFSD" and c["seed"] == seed]
        if len(contexts) != 1:
            raise ValueError("one original MSU-excluded context per fixed seed required")
        validate_role_isolation(contexts[0], rows, queried_hashes)
        path = comparison / "contexts" / f"MSU-MFSD-{seed}" / "models.json"
        validate_models(read_json(path, sha), seed)
        bindings[str(path)] = sha
    upstream = Path.home() / ".cache/fas/upstream/dinov2-7764ea0"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if revision != DINO_REVISION or dirty:
        raise ValueError("clean pinned official DINO source required")
    for path in sorted((upstream / "dinov2").rglob("*.py")):
        bindings[str(path)] = digest(path)
    for name in ("preprocessing", "frozen_features", "source_fitting"):
        path = fas_root / f"src/fas/{name}.py"
        bindings[str(path)] = digest(path)
    for branch, sha in zip(BRANCHES, WEIGHT_SHAS, strict=True):
        path = Path.home() / f".cache/fas/source-pilot295-v1/checkpoints/{branch}.pth"
        if digest(path) != sha:
            raise ValueError("pinned already acquired DINO weights required")
        bindings[str(path)] = sha
    for name in ("plan.json", "result.json", "supervision.json", "postflight.json"):
        bindings[str(mat_root / name)] = digest(mat_root / name)
    return mat, bindings


def build_plan(mat_root: Path, output: Path, approval: Path) -> Path:
    require_private_output(output / "plan.json")
    if output.exists():
        raise ValueError("new DINO comparison root required; no retry")
    mat = read_json(mat_root / "plan.json", MAT_SHA)
    _, bindings = input_bindings(mat_root, Path(mat["fas_root"]))
    accepted = read_json(approval)
    if (accepted["explicit_approval"] is not True or accepted["authority"] != AUTHORITY
            or accepted["budget"] != BUDGET or accepted["policy"] != POLICY
            or accepted["seeds"] != list(SEEDS)):
        raise PermissionError("explicit matched DINO scope, historical crop and budget approval required")
    bindings[str(approval.resolve())] = digest(approval)
    project = Path(__file__).resolve().parents[2]
    plan = {"scope": "MSU_matched_video_frozen_DINO_MAT_development_diagnostic", "authority": AUTHORITY,
            "policy": POLICY, "budget": BUDGET, "seeds": list(SEEDS), "entries": mat["entries"],
            "mat_root": str(mat_root.resolve()), "fas_root": mat["fas_root"], "bootstrap": mat["bootstrap"],
            "source_manifest": mat["source_manifest"], "decoder_config": mat["decoder_config"],
            "bindings": bindings, "implementation_sha256": {
                str(p.relative_to(project)): digest(p) for directory in ("src", "scripts")
                for p in sorted((project / directory).rglob("*.py"))}}
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def checked_inputs(path: Path, sha: str) -> dict[str, Any]:
    require_private_output(path)
    plan = read_json(path, sha)
    if (plan["scope"] != "MSU_matched_video_frozen_DINO_MAT_development_diagnostic"
            or plan["authority"] != AUTHORITY or plan["budget"] != BUDGET
            or plan["policy"] != POLICY or plan["seeds"] != list(SEEDS)):
        raise PermissionError("fixed approved frozen DINO protocol required")
    project = Path(__file__).resolve().parents[2]
    for name, expected in plan["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("DINO comparison implementation drift")
    for name, expected in plan["bindings"].items():
        if digest(Path(name)) != expected:
            raise ValueError("DINO comparison bound source/model/evidence drift")
    mat, bindings = input_bindings(Path(plan["mat_root"]), Path(plan["fas_root"]))
    if (any(plan["bindings"].get(k) != v for k, v in bindings.items())
            or any(plan[k] != mat[k] for k in ("entries", "source_manifest", "decoder_config", "bootstrap"))):
        raise ValueError("exact matched MAT originals, windows and decoder required")
    return plan


def strict_crop(rgb: np.ndarray, sample: dict[str, Any], preprocessing) -> tuple[np.ndarray, dict[str, Any]]:
    crop, metadata = preprocessing.shared_context_crop(rgb, np.asarray(sample["box"]), np.asarray(sample["landmarks"]))
    x1, y1, x2, y2 = metadata["crop_xyxy_half_open"]
    inverse = cv2.invertAffineTransform(np.asarray(metadata["alignment_matrix"]))
    corners = np.array([[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]])
    source = corners @ inverse[:, :2].T + inverse[:, 2]
    height, width = rgb.shape[:2]
    if (np.any(np.floor(source.min(axis=0)) - 1 < 0)
            or np.any(np.floor(source.max(axis=0)) + 2 > [width - 1, height - 1])):
        raise MediaFailure("DINO_legacy_cubic_footprint_outside_real_pixels")
    image = cv2.resize(crop, (518, 518), interpolation=cv2.INTER_CUBIC)
    return image, {**metadata, "native_width_height": [width, height],
                   "source_inverse_corners": source.tolist(), "cubic_support_real_pixels": True,
                   "rectangle_width_height": [x2 - x1, y2 - y1], "resized_to": [518, 518]}


def frame_scores(features: dict[str, np.ndarray], models: dict[str, Any], fitting) -> dict[str, dict[str, list[float]]]:
    if set(features) != set(BRANCHES) or any(
            v.shape != (16, 1536) or v.dtype != np.float32 or not np.isfinite(v).all() for v in features.values()):
        raise ValueError("both exact finite T16 native DINO feature profiles required")
    scores = {}
    for seed in SEEDS:
        model = models[str(seed)]
        validate_models(model, seed)
        p = {b: fitting.calibrate(model["calibrators"][b], fitting.head_logits(model["heads"][b], features[b]))
             for b in BRANCHES}
        scores[str(seed)] = {**{b: v.tolist() for b, v in p.items()},
                             "two_dino": ((p["dino_reg"] + p["dino_plain"]) / 2).tolist()}
    return scores


def pairwise_complementarity(dino: list[Prediction], mat: list[Prediction]) -> dict[str, Any]:
    if [(r.key, r.label) for r in dino] != [(r.key, r.label) for r in mat] or any(
            r.score is None for r in [*dino, *mat]):
        raise ValueError("identical complete ordered populations required for pairwise comparison")
    attack = [i for i, r in enumerate(dino) if r.label == "attack"]
    real = [i for i, r in enumerate(dino) if r.label == "bona_fide"]
    if not attack or not real:
        raise ValueError("both classes required")
    table = np.zeros((3, 3), dtype=int)
    for a in attack:
        for b in real:
            signs = [int(np.sign(rows[a].score - rows[b].score)) + 1 for rows in (dino, mat)]
            table[tuple(signs)] += 1
    return {"ordering": ["wrong", "tie", "correct"], "rows": "DINO", "columns": "MAT",
            "counts": table.tolist(), "attack_bonafide_pairs": len(attack) * len(real),
            "DINO_wrong_MAT_correct": int(table[0, 2]), "DINO_correct_MAT_wrong": int(table[2, 0]),
            "both_wrong": int(table[0, 0]), "both_correct": int(table[2, 2]),
            "dependent_pairs_not_independent_trials": True, "not_operating_point_rescue": True}


def extract_feature(model, rgb: np.ndarray, frozen) -> np.ndarray:
    tensor = torch.from_numpy(frozen.normalize_rgb(rgb, 518, tuple(POLICY["mean"]), tuple(POLICY["std"]))[None]).cuda()
    with torch.inference_mode():
        output = model.forward_features(tensor)
    feature = frozen.dino_pool(output["x_norm_clstoken"].cpu().numpy(), output["x_norm_patchtokens"].cpu().numpy())
    frozen.check_feature(feature, 1, 1536)
    return feature


def run_comparison(path: Path, sha: str) -> None:
    plan = checked_inputs(path, sha)
    root, fas_root = path.parent, Path(plan["fas_root"])
    if (root / "started.json").exists():
        raise ValueError("DINO comparison already attempted; no retry")
    write_json(root / "started.json", {"plan_sha256": sha})
    deadline = time.monotonic() + BUDGET["max_seconds"] - 20
    for name in ("records", "pixels", "staging", "features"):
        (root / name).mkdir()
    cv2.setNumThreads(1)
    modules = fas_modules(fas_root)
    decoders, helper_sha = decoder_policy(plan["decoder_config"])
    if decoders["MSU-MFSD"] != "fas_strict_ffmpeg" or helper_sha is None:
        raise ValueError("strict native replay required")
    native = FasNativeReplay(fas_root, helper_sha, Path(plan["source_manifest"]), MANIFEST_SHA)
    with ExitStack() as stack:
        context: dict[str, Any] = {}
        for index, entry in enumerate(plan["entries"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("DINO shared wall cap")
            video = item_video(entry)
            staged = root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}"
            write_json(root / "records" / f"started-{index:02d}.json", {"key": video.key})
            try:
                stage_member(entry, fas_root, fas_root, staged, stack, context, deadline, allow_validation_attacks=True)
                replay = native.bind(replace(video, media_path=str(staged)), root / "native-replay" / f"original-{index:02d}",
                                     expected_receipt_sha256=entry["receipt_sha256"])
                frames = read_native_frames(video, staged, {s["native_order"] for s in entry["samples"]},
                                            min(deadline, time.monotonic() + 180), replay)
                images, records = [], []
                for sample in entry["samples"]:
                    rgb, geometry = strict_crop(frames[sample["native_order"]], sample, modules["preprocessing"])
                    images.append(rgb)
                    records.append({"clip_slot": sample["clip_slot"], "geometry": geometry,
                                    "RGB_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                                    "native_RGB_sha256": video.frame_hashes[sample["native_order"]]})
                pixels = root / "pixels" / f"{index:02d}.npz"
                with pixels.open("xb") as stream:
                    np.savez_compressed(stream, crops=np.stack(images))
                write_json(root / "records" / f"crop-{index:02d}.json",
                           {"key": video.key, "frames": records, "pixels_sha256": digest(pixels)})
                del frames, images
            finally:
                staged.unlink(missing_ok=True)
            print({"phase": "DINO_native_crops", "originals": index + 1, "total": 22}, flush=True)
    write_json(root / "device.json", configure_mat_gpu(plan))
    upstream = Path.home() / ".cache/fas/upstream/dinov2-7764ea0"
    sys.path.insert(0, str(upstream))
    factories = importlib.import_module("dinov2.hub.backbones")
    if not Path(factories.__file__).resolve().is_relative_to(upstream):
        raise ValueError("unbound DINO architecture import")
    smoke, frozen_states = {}, {}
    for branch, factory in zip(BRANCHES, ("dinov2_vitb14_reg", "dinov2_vitb14"), strict=True):
        model = getattr(factories, factory)(pretrained=False)
        weight_path = Path.home() / f".cache/fas/source-pilot295-v1/checkpoints/{branch}.pth"
        model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
        model.requires_grad_(False).eval().cuda()
        state_sha = tensor_digest(model.state_dict())
        synthetic = np.full((518, 518, 3), 127, dtype=np.uint8)
        first = extract_feature(model, synthetic, modules["frozen_features"])
        second = extract_feature(model, synthetic, modules["frozen_features"])
        delta = modules["frozen_features"].parity(first, second)
        smoke[branch] = {"repeat_max_absolute_difference": delta, "rtol": 1e-5, "atol": 1e-6}
        for index, entry in enumerate(plan["entries"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("DINO shared wall cap")
            pixels = root / "pixels" / f"{index:02d}.npz"
            crop = read_json(root / "records" / f"crop-{index:02d}.json")
            if digest(pixels) != crop["pixels_sha256"]:
                raise ValueError("DINO crop cache drift")
            with np.load(pixels, allow_pickle=False) as cache:
                values = np.concatenate([extract_feature(model, rgb, modules["frozen_features"]) for rgb in cache["crops"]])
            with (root / "features" / f"{branch}-{index:02d}.npy").open("xb") as stream:
                np.save(stream, values, allow_pickle=False)
            print({"phase": branch, "originals": index + 1, "total": 22}, flush=True)
        if tensor_digest(model.state_dict()) != state_sha:
            raise ValueError("frozen DINO model state changed")
        frozen_states[branch] = state_sha
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(root / "smoke.json", smoke)
    comparison = fas_root.parent / "FAS-private/artifacts/phase1/paper_comparison323_v1"
    models = {str(seed): read_json(comparison / "contexts" / f"MSU-MFSD-{seed}" / "models.json") for seed in SEEDS}
    mat_plan = read_json(Path(plan["mat_root"]) / "plan.json", MAT_SHA)
    baseline = mat_predictions(mat_plan, Path(plan["mat_root"]))
    arms = {str(seed): {name: [] for name in ("dino_reg", "dino_plain", "two_dino")} for seed in SEEDS}
    for index, entry in enumerate(plan["entries"]):
        video = item_video(entry)
        features = {b: np.load(root / "features" / f"{b}-{index:02d}.npy", allow_pickle=False) for b in BRANCHES}
        scores = frame_scores(features, models, modules["source_fitting"])
        for seed in SEEDS:
            for name in arms[str(seed)]:
                score = float(np.asarray(scores[str(seed)][name], dtype=np.float64).mean())
                arms[str(seed)][name].append(Prediction(video.key, video.dataset, video.group, video.partition,
                                                        video.label, score, None, video.attack_family))
        write_json(root / "records" / f"score-{index:02d}.json", {"key": video.key, "frame_probabilities": scores})
    results = {}
    for seed in SEEDS:
        paired = {"MAT": baseline["MAT_primary"], **arms[str(seed)]}
        results[str(seed)] = {
            "summaries": {name: auroc_summary(rows) for name, rows in paired.items()},
            "paired_subject_bootstrap": paired_auroc_bootstrap(paired, **plan["bootstrap"]),
            "complementarity": pairwise_complementarity(paired["two_dino"], paired["MAT"])}
    write_json(root / "result.json", {
        "plan_sha256": sha, "originals": 22, "frames": 352, "encoder_image_forwards": 704,
        "synthetic_forwards": 4, "results": results, "frozen_states_before_and_after": frozen_states,
        "equal_seed_descriptive_AUROC": {
            name: float(np.mean([results[str(s)]["summaries"][name]["auroc_usable"] for s in SEEDS]))
            for name in ("dino_reg", "dino_plain", "two_dino")},
        "new_fits": 0, "new_download_bytes": 0, "new_calibration_or_target_pixel_access": 0,
        "new_threshold_or_fusion_selection": False, "independent_test_claimed": False,
        "files_sha256": {str(p.relative_to(root)): digest(p) for folder in ("records", "pixels", "features", "native-replay")
                         for p in sorted((root / folder).glob("*")) if p.is_file()}})


def summarize(path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    root, plan = path.parent, read_json(path)
    for index, entry in enumerate(plan["entries"]):
        (root / "staging" / f"source-{index:02d}{Path(entry['member']).suffix}").unlink(missing_ok=True)
    result = read_json(root / "result.json") if (root / "result.json").exists() else None
    if result is not None:
        if result["plan_sha256"] != digest(path):
            raise ValueError("DINO result plan drift")
        for name, expected in result["files_sha256"].items():
            if digest(root / name) != expected:
                raise ValueError("DINO evidence drift")
    return {"completed": bool(supervision["completed"] and result is not None),
            "plan_sha256": digest(path), "budget": BUDGET, "policy": POLICY, "authority": AUTHORITY,
            "supervision": supervision, "smoke": read_json(root / "smoke.json") if (root / "smoke.json").exists() else None,
            "result": {k: v for k, v in result.items() if k != "files_sha256"} if result is not None else None,
            "not_matched_crop": True, "development_subset_not_independent_test": True}
