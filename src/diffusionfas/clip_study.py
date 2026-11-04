"""Receipt-bound, one-attempt source-supervised short-clip DINO probe."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from fractions import Fraction
import gc
import importlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any

import cv2
import numpy as np
import torch

from .clip_dino import (
    ARMS, CONTRASTS, DIAGNOSTIC, DIMENSION, FRAMES, SEED, SETTINGS, SYSTEMS,
    balanced_weights, fit_head, logits, middle_window, shuffled_features,
)
from .data import addressed_seed
from .fas_import import read_media_receipt
from .gpu_probe import tensor_digest
from .io import digest, object_digest, read_json, require_private_output, write_json
from .native_replay import FAS_NATIVE_HELPER_SHA256
from .preprocessing import FasFaceProvider, MediaFailure, interpolate_track
from .regional_study import DINO_REVISION, inputs, paired_intervals, runtime, source_services, source_state
from .source_preparation import DETECTOR_SHA
from .source_probe import copy_member, safe_member, stage_archive_member

BUDGET = {"max_seconds": 24*3600, "max_rss_bytes": 8*1024**3,
          "max_gpu_bytes": 8*1024**3, "max_output_bytes": 4*1024**3}
POLICY = {
    "duration_seconds": 2, "fps": 8, "frames": FRAMES, "single_slot": FRAMES//2,
    "window": "native_interval_middle_uniform_bin_centers_nearest_unique_native_frames",
    "crop": "original_shared_context_crop_then_bicubic_RGB224_tracked_three_point_smoothing",
    "tracking": "original_interpolate_track_endpoints_required_max_gap2_jump0.5_area_ratio0.5_to2",
    "failures": "all_four_arms_shared_nonacceptance_original_denominators",
    "historical_31_failures": "retained_without_recovery",
    "shuffle": "fixed_addressed_nonidentity_permutation_per_target_original_all_roles",
    "inference_shuffle": "same_temporal_head_calibrator_and_chronological_source_policy_no_refit",
    "encoder": "frozen_fold_specific_completed_paper329_reference_normalized_CLS_RGB224",
    "primary_candidate": "temporal", "reference": "mean", "order_specific_control": "shuffled",
    "selection": SETTINGS["selection"], "target_model_selection": False,
    "primary_alpha": "1/20", "sensitivity_alphas": ["1/100", "1/10"],
    "bootstrap_repetitions": 2000, "bootstrap_seed": 20261028,
    "grouping": "subject_or_group_and_historical_or_any_selected_native_frame_connected_components",
    "new_cross_role_or_domain_frame_collision": "hard_stop_before_pixel_access",
    "automatic_retry": False, "automatic_expansion": False,
    "development_targets_already_seen": True, "SiW_subject_independence_verified": False,
}
SCOPE = {"targets": 4, "seed": SEED, "head_fits": 16, "calibrator_fits": 16,
         "operating_cases": 60, "conditional_intervals": 132,
         "encoder_training": False, "new_download": False, "new_environment": False,
         "source_and_target_video_decode": True, "source_only_head_selection_and_calibration": True,
         "retained_PAD_pixel_caches": False}
ANSWER = "approve_exact_clip_dino_v1_plan_no_retry"
ROLES = ("fitting", "validation", "calibration", "evaluation")


def code_pins() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)): digest(p) for d in ("src", "scripts")
            for p in sorted((root/d).rglob("*.py"))}


def clip_runtime() -> dict[str, str]:
    import onnxruntime

    return {**runtime(), "onnxruntime": onnxruntime.__version__,
            "ffmpeg": subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True,
                                     text=True, timeout=15).stdout.splitlines()[0]}


def file_identity(path: Path) -> list[int]:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("regular non-symlink archive required")
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def bind(files: dict[str, str], path: Path, expected: str | None = None) -> str:
    value = digest(path)
    if expected is not None and value != expected:
        raise ValueError(f"bound input changed: {path.name}")
    files[str(path.resolve())] = value
    return value


def checkpoint_bindings(root: Path, files: dict[str, str]) -> dict[str, Any]:
    terminal = read_json(root/"terminal.json")
    verification = read_json(root/"verification.json", terminal["verification_sha256"])
    completion = read_json(root/"completion.json", verification["completion_sha256"])
    if (terminal["status"] != "complete" or terminal["failure"] is not None
            or verification["contexts"] != 8 or verification["new_fits"] != 0
            or completion["scientific_training_starts"] != 8 or completion["calibrator_fits"] != 8):
        raise ValueError("complete independently reconstructed adapted checkpoint study required")
    for name in ("terminal.json", "verification.json", "completion.json", "execution_plan.json"):
        bind(files, root/name)
    result = {}
    for context in completion["contexts"]:
        target = context["target"]
        if target in result or context["seed"] != SEED:
            raise ValueError("one adapted reference checkpoint per held-out target/seed required")
        artifact = context["artifacts"]["reference"]
        folder = root/"contexts"/f"{target}-reference"
        bind(files, folder/"operating.json", artifact["operating"])
        checkpoint_sha = bind(files, folder/"model.pth", artifact["checkpoint"])
        saved = read_json(folder/"operating.json")
        if saved["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("adapted reference checkpoint/operating join changed")
        result[target] = {"path": str(folder/"model.pth"), "sha256": checkpoint_sha,
                          "selected_epoch": saved["selected_epoch"]}
    if len(result) != 4:
        raise ValueError("four source-only adapted reference checkpoints required")
    return result


def temporal_groups(rows: list[dict[str, Any]], entries: list[dict[str, Any]],
                    roles: dict[str, str]) -> dict[str, str]:
    parent = {entry["key"]: entry["key"] for entry in entries}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen: dict[tuple[str, ...], str] = {}
    for row, entry in zip(rows, entries, strict=True):
        key, domain = entry["key"], row["dataset"]
        tokens = [("group", domain, row["group_id"])]
        if row["subject_id"]:
            tokens.append(("subject", domain, row["subject_id"]))
        hashes = [frame["rgb_sha256"] for frame in entry.get("frames", [])]
        if row["selected_frame"] is not None:
            hashes.append(row["selected_frame"]["rgb_sha256"])
        tokens.extend(("frame", value) for value in hashes)
        for token in tokens:
            if token in seen:
                previous = seen[token]
                if roles[previous] != roles[key] or previous.split("/", 1)[0] != domain:
                    raise ValueError("new temporal exact-content/group collision crosses role or dataset")
                parent[find(key)] = find(previous)
            else:
                seen[token] = key
    groups: dict[str, list[str]] = {}
    for key in parent:
        groups.setdefault(find(key), []).append(key)
    return {key: object_digest(sorted(members)) for members in groups.values() for key in members}


def build_plan(fas_root: Path, output: Path, *, adapted_root: Path, oulu_root: Path,
               detector: Path, data_root: Path) -> Path:
    require_private_output(output/"plan.json")
    if output.exists():
        raise FileExistsError("new clip-probe root required")
    begun = time.monotonic()
    environment = clip_runtime()
    services, rows, partitions = inputs(fas_root)
    preparation = services["prepare_paper319"]
    contexts = [c for c in partitions["contexts"] if c["seed"] == SEED]
    roles: dict[str, str] = {}
    for context in contexts:
        for role in ROLES:
            for key in context[role]:
                if key in roles and roles[key] != role:
                    raise ValueError("fixed original role changed between target contexts")
                roles[key] = role
    files: dict[str, str] = {}
    bind(files, preparation.MANIFEST, preparation.MANIFEST_SHA)
    bind(files, preparation.OUTPUT/"partitions.json")
    bind(files, detector, DETECTOR_SHA)
    bind(files, fas_root/"scripts/msu_replay282.py", FAS_NATIVE_HELPER_SHA256)
    checkpoints = checkpoint_bindings(adapted_root, files)
    if set(checkpoints) != {c["target"] for c in contexts} or len(contexts) != 4:
        raise ValueError("exact four-target checkpoint/partition join required")
    entries, archives = [], set()
    for index, row in enumerate(rows):
        if time.monotonic()-begun > 1800:
            raise TimeoutError("metadata preparation 30-minute cap")
        key = services["fas.source_fitting"].identity(row)
        entry: dict[str, Any] = {"key": key, "dataset": row["dataset"], "role": roles[key],
                                 "label": row["binary_label"], "failure": None}
        if row["detector_status"] != "success":
            entry["failure"] = "historical_"+str(row["failure_kind"])
            entries.append(entry)
            continue
        bind(files, Path(row["record_path"]), row["record_sha256"])
        media, media_sha = read_media_receipt(row, preparation.PRIVATE/"oulu_per_video_audit_v1/videos")
        transaction_path = Path(row["record_path"])
        receipt_path = (preparation.PRIVATE/"oulu_per_video_audit_v1/videos"/f"{row['video_id']}.json"
                        if row["dataset"] == "OULU-NPU"
                        else transaction_path.parent.parent/"media"/transaction_path.name)
        bind(files, receipt_path, media_sha)
        entry.update(receipt=str(receipt_path), receipt_sha256=media_sha,
                     member=media["media_relpath"], byte_size=media["byte_size"],
                     media_sha256=media["media_sha256"], native_frame_count=media["frame_count"])
        if media["decode_status"] != "decoded" or media["frame_count"] != len(media["frame_index"]):
            raise ValueError("accepted usable original lacks canonical decoded media receipt")
        if not 0 < media["byte_size"] <= 128*1024**2:
            entry["failure"] = "media_exceeds_fixed_128MiB_staging_cap"
        else:
            timestamps = tuple(float(Fraction(f["timestamp_seconds"])) for f in media["frame_index"])
            try:
                selected = middle_window(timestamps)
            except ValueError as error:
                if str(error) not in {"insufficient_duration", "distinct native frames required; no duplication/interpolation",
                                     "frame exceeds half-sampling-period tolerance"}:
                    raise
                entry["failure"] = str(error)
            else:
                entry["frames"] = [media["frame_index"][i] for i in selected]
                if len({(f["width"], f["height"]) for f in media["frame_index"]}) != 1:
                    entry["failure"] = "variable_native_dimensions"
                elif sum(f["width"]*f["height"]*3 for f in entry["frames"]) > 1024**3:
                    entry["failure"] = "selected_native_frame_memory_exceeds_1GiB"
        if row["dataset"] == "OULU-NPU":
            archive = oulu_root/media["media_archive"]
            entry["archive"] = str(archive)
            archives.add(archive)
        elif row["dataset"] == "CASIA-FASD":
            archives.add(data_root/"casia-fasd/CASIA_faceAntisp.rar")
        elif row["dataset"] == "SiW-Mv2":
            archives.add(data_root/"SiW/SiW-Mv2.zip")
        else:
            archives.update((data_root/"MSU").glob("*.zip"))
        entries.append(entry)
        if index % 1000 == 0:
            print({"phase": "clip_metadata", "originals": index+1}, flush=True)
    groups = temporal_groups(rows, entries, roles)
    eligible = {entry["key"] for entry in entries if entry["failure"] is None}
    indexed = {services["fas.source_fitting"].identity(r): r for r in rows}
    inventory = []
    for context in contexts:
        for role in ROLES[:3]:
            pool = [indexed[key] for key in context[role] if key in eligible]
            balanced_weights(np.asarray([r["binary_label"] == "attack" for r in pool], dtype=int),
                             [r["dataset"] for r in pool])
        keys = sorted({key for role in ROLES for key in context[role]})
        inventory.append({"target": context["target"], "originals": len(keys),
                          "metadata_eligible": sum(key in eligible for key in keys),
                          "maximum_encoder_frames": FRAMES*sum(key in eligible for key in keys),
                          "role_counts": {
                              role: {"originals": len(context[role]),
                                     "metadata_eligible": sum(key in eligible for key in context[role]),
                                     "usable_by_domain_class": dict(Counter(
                                         indexed[key]["dataset"]+"/"+indexed[key]["binary_label"]
                                         for key in context[role] if key in eligible))}
                              for role in ROLES}})
    estimate = sum(r["originals"] for r in inventory)*FRAMES*DIMENSION*4+512*1024**2
    if estimate > BUDGET["max_output_bytes"] or not archives:
        raise ValueError("bounded feature/staging estimate and existing archive inventory required")
    if os.statvfs(output.parent).f_bavail*os.statvfs(output.parent).f_frsize < estimate+1024**3:
        raise OSError("insufficient free native output storage")
    archive_identities = {str(path): file_identity(path) for path in sorted(archives)}
    plan = {"scope": SCOPE, "policy": POLICY, "settings": SETTINGS, "budget": BUDGET,
            "fas_root": str(fas_root.resolve()), "data_root": str(data_root.resolve()),
            "detector": str(detector.resolve()), "checkpoints": checkpoints,
            "upstream": str(Path.home()/".cache/fas/upstream/dinov2-7764ea0"),
            "implementation_sha256": code_pins(), "runtime": environment, "files": files,
            "archives": archive_identities, "entries": entries, "contexts": contexts, "groups": groups,
            "inventory": inventory, "conservative_owned_output_estimate_bytes": estimate,
            "metadata_failure_counts": dict(Counter(e["failure"] for e in entries if e["failure"])),
            "maximum_full_native_frames": sum(e.get("native_frame_count", 0) for e in entries if not e["failure"]),
            "maximum_detector_calls": FRAMES*len(eligible), "preparation_wall_seconds": time.monotonic()-begun}
    output.mkdir(parents=True)
    write_json(output/"plan.json", plan)
    return output/"plan.json"


def checked_plan(path: Path, sha: str) -> dict[str, Any]:
    require_private_output(path)
    plan = read_json(path, sha)
    if (plan["scope"] != SCOPE or plan["policy"] != POLICY or plan["budget"] != BUDGET
            or plan["settings"] != SETTINGS or plan["implementation_sha256"] != code_pins()
            or plan["runtime"] != clip_runtime()):
        raise ValueError("sealed clip scope, recipe, implementation or runtime changed")
    return plan


def verify_inputs(plan: dict[str, Any], guard: Callable[[], None]) -> None:
    source_state(Path(plan["fas_root"]))
    for name, sha in plan["files"].items():
        guard()
        if digest(Path(name)) != sha:
            raise ValueError(f"clip input changed: {Path(name).name}")
    for name, expected in plan["archives"].items():
        guard()
        if file_identity(Path(name)) != expected:
            raise ValueError("clip archive identity changed")
    upstream = Path(plan["upstream"])
    for args, expected in ((["rev-parse", "HEAD"], DINO_REVISION), (["status", "--porcelain"], "")):
        observed = subprocess.run(["git", "-C", str(upstream), *args], check=True,
                                  capture_output=True, text=True, timeout=15).stdout.strip()
        if observed != expected:
            raise ValueError("pinned clean DINO upstream required")


def require_approval(path: Path, sha: str, receipt: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = checked_plan(path, sha)
    approval = read_json(receipt)
    if (approval["answer"] != ANSWER or approval["plan_sha256"] != sha
            or approval["scope"] != SCOPE or approval["budget"] != BUDGET
            or approval["policy"] != POLICY or approval["automatic_retry"] is not False
            or approval["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()):
        raise PermissionError("explicit exact-plan owner approval and original boot clock required")
    remaining = approval["approved_monotonic"]+BUDGET["max_seconds"]-time.monotonic()
    if remaining <= 30:
        raise TimeoutError("nonresettable clip approval clock expired")
    return plan, approval


def prepare_clip(frames: list[np.ndarray], provider: FasFaceProvider,
                 guard: Callable[[], None]) -> tuple[np.ndarray, dict[str, Any]]:
    if len(frames) != FRAMES:
        raise ValueError("exactly16 canonical native frames required")
    detections = []
    for rgb in frames:
        guard()
        detections.append(provider.detect(rgb))
    boxes, points = interpolate_track(detections)
    crops, geometry = [], []
    for rgb, box, landmarks in zip(frames, boxes, points, strict=True):
        guard()
        try:
            crop, description = provider.module.shared_context_crop(rgb, box, landmarks)
        except ValueError as error:
            if str(error) not in {"degenerate facial landmarks", "invalid similarity transform",
                                  "positive finite face box required", "face context outside image"}:
                raise
            raise MediaFailure("unsupported_clip_crop:"+str(error)) from error
        crops.append(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC))
        geometry.append(description)
    return np.stack(crops), {"boxes": boxes.tolist(), "landmarks": points.tolist(), "geometry": geometry}


def stage(entry: dict[str, Any], plan: dict[str, Any], output: Path, stack: ExitStack,
          context: dict[str, Any], deadline: float) -> None:
    if entry["failure"] is not None:
        raise ValueError("metadata-ineligible video cannot be staged")
    if entry["dataset"] == "OULU-NPU":
        name = entry["archive"]
        if name not in context:
            context[name] = stack.enter_context(tarfile.open(name, "r:"))
        archive = context[name]
        safe_member(entry["member"])
        members = [member for member in archive.getmembers() if member.name == entry["member"]]
        if len(members) != 1 or not members[0].isfile() or members[0].size != entry["byte_size"]:
            raise ValueError("unique regular exact OULU tar member required")
        source = archive.extractfile(members[0])
        if source is None:
            raise ValueError("OULU member stream unavailable")
        with source:
            copy_member(source, output, entry["byte_size"], entry["media_sha256"], deadline)
    else:
        item = {**entry, "video": {"dataset": entry["dataset"], "media_sha256": entry["media_sha256"]}}
        stage_archive_member(item, Path(plan["fas_root"]), Path(plan["data_root"]), output,
                             stack, context, deadline)


def extract(plan: dict[str, Any], root: Path, guard: Callable[[], None]) -> dict[str, Any]:
    from .mat_preview import configure_mat_gpu
    write_json(root/"device.json", configure_mat_gpu(plan))
    sys.path.insert(0, plan["upstream"])
    factories = importlib.import_module("dinov2.hub.backbones")
    if not Path(factories.__file__).resolve().is_relative_to(Path(plan["upstream"])):
        raise ValueError("unbound DINO factory import")
    services, _, _ = inputs(Path(plan["fas_root"]))
    normalize = services["fas.frozen_features"].normalize_rgb
    models, arrays, positions, state_hashes = {}, {}, {}, {}
    for target, binding in plan["checkpoints"].items():
        guard()
        saved = torch.load(binding["path"], map_location="cpu", weights_only=True)
        if saved["seed"] != SEED or saved["tail"] is not False or saved["selected_epoch"] != binding["selected_epoch"]:
            raise ValueError("adapted reference checkpoint provenance changed")
        model = factories.dinov2_vitb14_reg(pretrained=False)
        state = {k.removeprefix("backbone."): v for k, v in saved["state_dict"].items() if k.startswith("backbone.")}
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False).cuda()
        models[target] = model
        state_hashes[target] = tensor_digest(model.state_dict())
        del saved, state
        context = next(c for c in plan["contexts"] if c["target"] == target)
        keys = sorted({key for role in ROLES for key in context[role]})
        positions[target] = {key: i for i, key in enumerate(keys)}
        arrays[target] = np.lib.format.open_memmap(root/f"{target}.npy", mode="w+", dtype=np.float32,
                                                  shape=(len(keys), FRAMES, DIMENSION))
        arrays[target][:] = np.nan
    provider = FasFaceProvider(Path(plan["fas_root"]),
                               digest(Path(plan["fas_root"])/"src/fas/preprocessing.py"),
                               Path(plan["detector"]), DETECTOR_SHA)
    sys.path.insert(0, str(Path(plan["fas_root"])/"scripts"))
    native = importlib.import_module("msu_replay282")
    if Path(native.__file__).resolve() != Path(plan["fas_root"])/"scripts/msu_replay282.py":
        raise ValueError("unbound native replay helper")
    synthetic = np.random.default_rng(SEED).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    for model in models.values():
        tensor = torch.from_numpy(normalize(synthetic, 224, (.485, .456, .406), (.229, .224, .225))[None]).cuda()
        with torch.inference_mode():
            first = model.forward_features(tensor)["x_norm_clstoken"]
            second = model.forward_features(tensor)["x_norm_clstoken"]
        if not torch.equal(first, second):
            raise ValueError("adapted encoder synthetic CLS replay failed")
    (root/"staging").mkdir()
    statuses, encoded, detector_calls, native_frames = {}, 0, 0, 0
    decode_hashes, tracking_hashes = {}, {}
    with ExitStack() as stack:
        archive_context: dict[str, Any] = {}
        for index, entry in enumerate(plan["entries"]):
            guard()
            key = entry["key"]
            if entry["failure"]:
                statuses[key] = {"success": False, "failure": entry["failure"]}
                continue
            for name, expected in plan["archives"].items():
                if file_identity(Path(name)) != expected:
                    raise ValueError("archive changed during clip extraction")
            media = read_json(Path(entry["receipt"]), entry["receipt_sha256"])
            path = root/"staging"/f"{index:05d}.media"
            deadline = time.monotonic()+180
            stage(entry, plan, path, stack, archive_context, deadline)
            wanted = {f["decode_order"] for f in entry["frames"]}
            observed: dict[int, np.ndarray] = {}

            def consume(order: int, rgb: np.ndarray) -> None:
                guard()
                if order in wanted:
                    observed[order] = rgb.copy()

            native.strict_replay(path, media, consume, root/"decode"/f"{index:05d}.json",
                                 timeout=max(.01, deadline-time.monotonic()))
            decode_hashes[f"{index:05d}"] = digest(root/"decode"/f"{index:05d}.json")
            native_frames += media["frame_count"]
            if set(observed) != wanted:
                raise ValueError("selected native frame coverage mismatch")
            frames = [observed[f["decode_order"]] for f in entry["frames"]]
            try:
                crops, tracking = prepare_clip(frames, provider, guard)
            except MediaFailure as error:
                statuses[key] = {"success": False, "failure": str(error)}
            else:
                value = np.stack([normalize(rgb, 224, (.485, .456, .406), (.229, .224, .225)) for rgb in crops])
                with torch.inference_mode():
                    tensor = torch.from_numpy(value).cuda()
                    for target, model in models.items():
                        if key in positions[target]:
                            guard()
                            features = model.forward_features(tensor)["x_norm_clstoken"].cpu().numpy()
                            if features.shape != (FRAMES, DIMENSION) or not np.isfinite(features).all():
                                raise ValueError("finite exact16x768 adapted CLS features required")
                            arrays[target][positions[target][key]] = features
                            encoded += FRAMES
                statuses[key] = {"success": True}
                write_json(root/"tracking"/f"{index:05d}.json", {"key": key, **tracking})
                tracking_hashes[f"{index:05d}"] = digest(root/"tracking"/f"{index:05d}.json")
                del crops, value, tensor
            detector_calls += FRAMES
            path.unlink()
            errors = path.with_suffix(path.suffix+".stderr")
            if errors.exists():
                errors.unlink()
            del observed, frames
            if index % 50 == 0:
                print({"phase": "clip_extraction", "originals": index+1, "encoder_frames": encoded}, flush=True)
    for target, model in models.items():
        guard()
        if tensor_digest(model.state_dict()) != state_hashes[target]:
            raise ValueError("frozen adapted encoder changed")
        arrays[target].flush()
    arrays.clear()
    models.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    receipt = {"statuses": statuses, "positions": positions, "encoder_state_sha256": state_hashes,
               "files": {target: digest(root/f"{target}.npy") for target in positions},
               "production_encoder_frames": encoded, "synthetic_encoder_frames": 8,
               "detector_calls_upper_bound": detector_calls, "canonical_native_frames_verified": native_frames,
               "exact_synthetic_replay": True, "decode_sha256": decode_hashes,
               "tracking_sha256": tracking_hashes}
    write_json(root/"features.json", receipt)
    return receipt


def study_rows(rows: list[dict[str, Any]], receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = f"{row['dataset']}/{row['video_id']}"
        status = receipt["statuses"][key]
        result[key] = (row if status["success"] else {
            **row, "detector_status": "failed", "failure_kind": status["failure"], "final_k1_action": "non_accept"})
    return result


def source_pools(context: dict[str, Any], indexed: dict[str, dict[str, Any]],
                 positions: dict[str, int], array: np.ndarray) -> tuple[dict, dict, dict]:
    pools = {role: [indexed[key] for key in context[role] if indexed[key]["detector_status"] == "success"]
             for role in ROLES}
    keys = {role: [f"{r['dataset']}/{r['video_id']}" for r in pool] for role, pool in pools.items()}
    features = {role: np.asarray(array[[positions[key] for key in keys[role]]]) for role in ROLES}
    for role in ROLES[:3]:
        balanced_weights(np.asarray([r["binary_label"] == "attack" for r in pools[role]], dtype=int),
                         [r["dataset"] for r in pools[role]])
    return pools, keys, features


def analyze(target: str, predictions: dict[str, np.ndarray], policies: dict[str, Any],
            originals: list[dict[str, Any]], services: dict[str, Any], guard: Callable[[], None],
            group_map: dict[str, str]) -> tuple[list, list]:
    study = services["study"]
    cases = []
    for arm in SYSTEMS:
        positions, labels = study.score_inputs(predictions[arm], originals)
        for alpha in study.ALPHAS:
            point = study.metrics(predictions[arm], originals, policies[arm][str(alpha)]["threshold"])
            point["FA_within_constraint"] = point["accepted_attack"] <= int(point["original_attack"])*alpha.numerator//alpha.denominator
            point["utility_feasible"] = point["FA_within_constraint"] and point["BFA"] >= .5
            oracle = services["oracle316"].oracle(
                predictions[arm], labels[positions], int(labels.sum()), int((1-labels).sum()), alpha=alpha)
            cases.append({"target": target, "seed": SEED, "arm": arm, "alpha": str(alpha),
                          "diagnostic_only": arm == DIAGNOSTIC, "policy": point,
                          "oracle": oracle})
    intervals = paired_intervals(predictions, originals, policies, services, guard,
                                 arms=SYSTEMS, contrasts=CONTRASTS, group_map=group_map)
    return cases, [{"target": target, "seed": SEED, **r} for r in intervals]


def run_worker(path: Path, sha: str, approval_path: Path) -> None:
    plan, approval = require_approval(path, sha, approval_path)
    root = path.parent
    attempt = read_json(root/"attempt.json")
    if attempt["supervisor_pid"] != os.getppid() or attempt["plan_sha256"] != sha:
        raise PermissionError("clip worker requires its recorded supervisor")
    write_json(root/"started.json", {"worker_pid": os.getpid(), "plan_sha256": sha,
                                   "approval_sha256": digest(approval_path)})
    deadline = approval["approved_monotonic"]+BUDGET["max_seconds"]-30
    supervisor = os.getppid()

    def guard() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("clip shared approval wall-clock cap")
        if os.getppid() != supervisor:
            raise RuntimeError("clip worker lost supervisor")

    verify_inputs(plan, guard)
    services, rows, partitions = inputs(Path(plan["fas_root"]))
    if [c for c in partitions["contexts"] if c["seed"] == SEED] != plan["contexts"]:
        raise ValueError("clip context inventory changed")
    receipt = extract(plan, root, guard)
    indexed = study_rows(rows, receipt)
    fitting, study = services["fas.source_fitting"], services["study"]
    cases, intervals, artifacts = [], [], {}
    for context in plan["contexts"]:
        target = context["target"]
        array = np.load(root/f"{target}.npy", mmap_mode="r", allow_pickle=False)
        pools, keys, features = source_pools(context, indexed, receipt["positions"][target], array)
        study.isolation([[indexed[key] for key in context[role]] for role in ROLES])
        originals = [indexed[key] for key in context["evaluation"]]
        group_map = {key: plan["groups"][key] for key in context["evaluation"]}
        labels = {role: np.asarray([r["binary_label"] == "attack" for r in pools[role]], dtype=int)
                  for role in ROLES}
        domains = {role: [r["dataset"] for r in pools[role]] for role in ROLES}
        shuffled = {role: shuffled_features(features[role], keys[role], target) for role in ROLES}
        predictions, policies, models = {}, {}, {}
        for arm in ARMS:
            guard()
            folder = root/"heads"/f"{target}-{arm}"
            write_json(folder/"started.json", {"target": target, "arm": arm, "seed": SEED, "plan_sha256": sha})
            selected = shuffled if arm == "shuffled" else features
            model, state = fit_head(arm, selected["fitting"], labels["fitting"], domains["fitting"],
                                    selected["validation"], labels["validation"], domains["validation"], guard,
                                    epochs=SETTINGS["epochs"])
            with (folder/"head.pth").open("xb") as stream:
                torch.save(state, stream)
            raw = {role: logits(model, selected[role], guard) for role in ("calibration", "evaluation")}
            write_json(folder/"calibrator-started.json", {"plan_sha256": sha})
            calibrator = fitting.fit_calibrator(raw["calibration"], labels["calibration"], domains["calibration"], guard)
            probabilities = {role: fitting.calibrate(calibrator, value) for role, value in raw.items()}
            policies[arm] = {str(alpha): study.threshold(probabilities["calibration"], pools["calibration"], alpha)
                             for alpha in study.ALPHAS}
            predictions[arm] = probabilities["evaluation"]
            models[arm] = {"calibrator": calibrator, "policies": policies[arm],
                           "probabilities": {role: value.tolist() for role, value in probabilities.items()},
                           "head_sha256": digest(folder/"head.pth"), "selected_epoch": state["selected_epoch"],
                           "initial_state_sha256": tensor_digest(state["initial_state"])}
            if arm == "temporal":
                predictions[DIAGNOSTIC] = fitting.calibrate(calibrator, logits(model, shuffled["evaluation"], guard))
                policies[DIAGNOSTIC] = policies[arm]
            print({"phase": "clip_head_complete", "target": target, "arm": arm,
                   "selected_epoch": state["selected_epoch"]}, flush=True)
        if models["temporal"]["initial_state_sha256"] != models["shuffled"]["initial_state_sha256"]:
            raise ValueError("ordered/shuffled heads must have identical initial parameters")
        record = {"target": target, "keys": keys, "models": models, "policies": policies,
                  "prediction": {arm: value.tolist() for arm, value in predictions.items()}}
        output = root/f"context-{target}.json"
        write_json(output, record)
        artifacts[target] = digest(output)
        points, uncertainty = analyze(target, predictions, policies, originals, services, guard, group_map)
        cases.extend(points)
        intervals.extend(uncertainty)
        del features, shuffled, array
        gc.collect()
    write_json(root/"result.json", {"plan_sha256": sha, "features_sha256": digest(root/"features.json"),
                                  "artifacts": artifacts, "cases": cases, "conditional_intervals": intervals})
    guard()
    script = Path(__file__).resolve().parents[2]/"scripts/clip_dino.py"
    subprocess.run([sys.executable, str(script), "_verify", "--plan", str(path),
                    "--plan-sha256", sha, "--approval", str(approval_path)], check=True,
                   timeout=max(.01, deadline-time.monotonic()))


def preparation_summary(plan: dict[str, Any], sha: str) -> dict[str, Any]:
    return {"plan_sha256": sha, "scope": SCOPE, "policy": POLICY, "settings": SETTINGS, "budget": BUDGET,
            "scientific_execution_authorized": False, "new_PAD_video_decodes": 0,
            "new_PAD_model_forwards": 0, "new_PAD_head_fits": 0,
            "originals": len(plan["entries"]), "metadata_failure_counts": plan["metadata_failure_counts"],
            "metadata_eligible": sum(e["failure"] is None for e in plan["entries"]),
            "maximum_full_native_frames": plan["maximum_full_native_frames"],
            "maximum_detector_calls": plan["maximum_detector_calls"], "inventory": plan["inventory"],
            "conservative_owned_output_estimate_bytes": plan["conservative_owned_output_estimate_bytes"],
            "bound_files": len(plan["files"]), "bound_archives": len(plan["archives"]),
            "adapted_checkpoint_sha256": {t: c["sha256"] for t, c in plan["checkpoints"].items()},
            "preparation_wall_seconds": plan["preparation_wall_seconds"]}


def completed_summary(plan: dict[str, Any], root: Path, sha: str) -> dict[str, Any]:
    verification = read_json(root/"verification.json")
    supervision = read_json(root/"supervision.json")
    if (verification["completed"] is not True or supervision["completed"] is not True
            or verification["plan_sha256"] != sha or supervision["plan_sha256"] != sha):
        raise ValueError("successful bound supervision and reconstruction required")
    result = read_json(root/"result.json", verification["result_sha256"])
    features = read_json(root/"features.json", result["features_sha256"])
    cases = [{**case, "oracle": {k: v for k, v in case["oracle"].items() if k not in {
        "threshold_interval", "oracle_threshold_not_deployable"}}} for case in result["cases"]]
    means = []
    for arm in SYSTEMS:
        points = [case["policy"] for case in cases if case["arm"] == arm and case["alpha"] == "1/20"]
        if len(points) != 4:
            raise ValueError("four targets required in every primary aggregate")
        means.append({"arm": arm, "diagnostic_only": arm == DIAGNOSTIC,
                      "equal_target_macro": {metric: float(np.mean([p[metric] for p in points]))
                                             for metric in ("AUROC_usable", "FA", "BFA")},
                      "FA_safe_targets": sum(p["FA_within_constraint"] for p in points),
                      "joint_utility_targets": sum(p["utility_feasible"] for p in points)})
    return {"completed": True, "plan_sha256": sha, "scope": SCOPE, "policy": POLICY,
            "settings": SETTINGS, "budget": BUDGET, "seed_count": 1,
            "cases": cases, "conditional_intervals": result["conditional_intervals"],
            "primary_equal_target_summaries": means, "verification": verification,
            "resources": supervision, "retained_originals": len(plan["entries"]),
            "usable_clip_originals": sum(s["success"] for s in features["statuses"].values()),
            "failure_counts": dict(Counter(s["failure"] for s in features["statuses"].values() if not s["success"])),
            "production_encoder_frames": features["production_encoder_frames"],
            "canonical_native_frames_verified": features["canonical_native_frames_verified"],
            "adapted_checkpoint_sha256": {t: c["sha256"] for t, c in plan["checkpoints"].items()},
            "limitation": "development_pilot_one_seed_no_untouched_confirmation_no_Q3_or_novelty_claim"}


def cpu_smoke(fas_root: Path, adapted_root: Path) -> dict[str, Any]:
    """Engineering-only local adapted-weight forwards on generated RGB, never PAD pixels."""
    begun = time.monotonic()
    source_state(fas_root)
    services = source_services(fas_root)
    bindings = checkpoint_bindings(adapted_root, {})
    upstream = Path.home()/".cache/fas/upstream/dinov2-7764ea0"
    from fas.environment import verify_source_checkout
    verify_source_checkout(upstream, DINO_REVISION)
    sys.path.insert(0, str(upstream))
    factories = importlib.import_module("dinov2.hub.backbones")
    if not Path(factories.__file__).resolve().is_relative_to(upstream):
        raise ValueError("unbound DINO factory in CPU smoke")
    rgb = np.random.default_rng(SEED).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    x = torch.from_numpy(services["fas.frozen_features"].normalize_rgb(
        rgb, 224, (.485, .456, .406), (.229, .224, .225))[None])
    expected = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1)[None]/255
    expected = (expected-expected.new_tensor([.485, .456, .406])[None, :, None, None])/expected.new_tensor(
        [.229, .224, .225])[None, :, None, None]
    error = float(torch.max(torch.abs(x-expected)))
    if error > 1e-6:
        raise ValueError("adapted reference RGB224 normalization mismatch")
    records = []
    for target, binding in bindings.items():
        if time.monotonic()-begun > 600:
            raise TimeoutError("CPU synthetic smoke10-minute cap")
        saved = torch.load(binding["path"], map_location="cpu", weights_only=True)
        if saved["seed"] != SEED or saved["tail"] is not False:
            raise ValueError("CPU smoke requires the declared adapted reference")
        model = factories.dinov2_vitb14_reg(pretrained=False).eval().requires_grad_(False)
        model.load_state_dict({k.removeprefix("backbone."): v for k, v in saved["state_dict"].items()
                               if k.startswith("backbone.")}, strict=True)
        before = tensor_digest(model.state_dict())
        with torch.inference_mode():
            first = model.forward_features(x)["x_norm_clstoken"]
            second = model.forward_features(x)["x_norm_clstoken"]
        if (first.shape != (1, DIMENSION) or not torch.isfinite(first).all()
                or not torch.equal(first, second) or tensor_digest(model.state_dict()) != before
                or any(p.requires_grad for p in model.parameters())):
            raise ValueError("adapted frozen CPU feature/replay smoke failed")
        records.append({"target": target, "checkpoint_sha256": binding["sha256"],
                        "frozen_encoder_state_sha256": before, "shape": list(first.shape),
                        "exact_synthetic_replay": True, "trainable_encoder_parameters": 0})
        del model, saved, first, second
        gc.collect()
    return {"completed": True, "device": "cpu", "synthetic_encoder_frames": 8,
            "PAD_pixel_decodes": 0, "PAD_fits": 0, "normalization_max_absolute_error": error,
            "records": records, "wall_seconds": time.monotonic()-begun}
