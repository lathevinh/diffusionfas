"""Metadata preparation and bounded synthetic-only resource evidence."""

from __future__ import annotations

from collections import Counter
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

from .gpu_probe import tensor_digest
from .io import digest, object_digest, read_json, require_private_output, write_json
from .regional_dino import RegionSupportError
from .regional_study import inputs, runtime, source_state, transaction_for
from .spatial_dino import (
    ARMS, SEED, SETTINGS, SpatialReference, check_gradients, optimizer_for,
    paired_batch, patch_prior, points224, prior_image, source_modules,
)

GIB = 1024**3
PROBE_CAPS = {"max_seconds": 600, "max_rss_bytes": 8*GIB,
              "max_gpu_bytes": 12*GIB, "max_output_bytes": 2*GIB}
PROPOSED_LIMITS = {"max_seconds": 120*3600, "max_rss_bytes": 8*GIB,
                   "max_gpu_bytes": 12*GIB, "max_output_bytes": 16*GIB}
POLICY = {
    "source_training_domains": 3, "held_out_targets": 4, "seed": SEED,
    "arms": list(ARMS), "scientific_fits_proposed": 20, "calibrator_fits_proposed": 20,
    "pretrained_initialization": "original_generic_DINO_no_prior_PAD_finetuned_checkpoint",
    "input": "existing_single_middle_frame_RGB224_not_short_clip",
    "shared_failures": "original_31_plus_any_invalid_prior_geometry_all_five_arms",
    "prior_support": "positive_soft_Gaussian_truncation_at_crop_edges_no_full_ellipse_requirement",
    "augmentation": "pinned_reference_RGB_label_PDA_parity_plus_prior_flip_rotation_donor_transport",
    "threshold": "unchanged_minimum_source_attack_quantile_strict_score_less_than_threshold",
    "primary_alpha": "1/20", "sensitivity_alphas": ["1/100", "1/10"],
    "target_model_selection": False, "target_driven_prior_selection": False,
    "development_targets_already_seen": True, "SiW_subject_independence_verified": False,
    "new_download": False, "new_environment": False, "new_decoder_or_detector": False,
    "automatic_retry": False, "automatic_expansion": False,
    "current_authority": "metadata_and_synthetic_only_no_scientific_execution_authority",
}


def code_pins(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path)
            for path in [*sorted((root/"src").rglob("*.py")), *sorted((root/"scripts").glob("*.py")),
                         *sorted((root/"tests").glob("test_spatial*.py"))]}


def fixture() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows = [{"dataset": domain, "binary_label": label, "index": i}
            for i, (domain, label) in enumerate((domain, label)
                for domain in ("synthetic-A", "synthetic-B", "synthetic-C")
                for label in ("bona_fide", "attack"))]
    images = np.random.default_rng(329).integers(0, 256, (6, 224, 224, 3), dtype=np.uint8)
    points = np.array([[76, 80], [148, 80], [112, 115], [86, 151], [138, 151]], dtype=float)
    priors = np.stack([prior_image(points+np.array([i-3, 0])) for i in range(6)])
    return rows, images, priors


def synthetic_replay(checkpoint: Path, expected: str, root: Path) -> dict[str, Any]:
    if not checkpoint.parent.name.startswith("spatial-synthetic-") or digest(checkpoint) != expected:
        raise ValueError("bound temporary synthetic checkpoint required")
    source_state(root)
    source = source_modules(root)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["arm"] not in ARMS or state["settings_sha256"] != object_digest(SETTINGS):
        raise ValueError("synthetic checkpoint arm/recipe mismatch")
    source.model.configure(SEED)
    torch.cuda.set_per_process_memory_fraction(PROBE_CAPS["max_gpu_bytes"]/
                                               torch.cuda.get_device_properties(0).total_memory)
    model = SpatialReference(source.model.Reference(initialize=False), state["arm"]).cuda().eval()
    model.load_state_dict(state["state_dict"], strict=True)
    _, images, priors = fixture()
    with torch.no_grad():
        values = model(source.model.tensor_images(images, "cuda"),
                       torch.from_numpy(patch_prior(priors)).cuda())[0].cpu()
    return {"arm": state["arm"], "logits": values.tolist()}


def synthetic_probe(output: Path, fas_root: Path) -> None:
    if output.exists():
        raise FileExistsError("immutable synthetic probe output already exists")
    source_state(fas_root)
    source = source_modules(fas_root)
    if digest(source.model.CHECKPOINT) != "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71":
        raise ValueError("pinned generic pretrained DINO checkpoint required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the proposed GPU resource measurement")
    torch.cuda.set_per_process_memory_fraction(PROBE_CAPS["max_gpu_bytes"]/
                                               torch.cuda.get_device_properties(0).total_memory)
    rows, images, priors = fixture()
    load_rgb = lambda row: images[row["index"]].copy()
    load_prior = lambda row: priors[row["index"]].copy()
    started, results, initial_reference, common_initial, common_traces = time.monotonic(), [], None, None, None
    initial_logits = None

    def guard() -> None:
        if time.monotonic()-started > PROBE_CAPS["max_seconds"]:
            raise TimeoutError("synthetic probe wall cap")

    with tempfile.TemporaryDirectory(prefix="spatial-synthetic-", dir=output.parent) as directory:
        folder = Path(directory)
        for arm in ARMS:
            for cached in (source.augmentation.background, source.augmentation.blue, source.augmentation.moire):
                cached.cache_clear()
            source.model.configure(SEED)
            model = SpatialReference(source.model.Reference(), arm).cuda()
            current = tensor_digest(model.reference.state_dict())
            if initial_reference is not None and current != initial_reference:
                raise ValueError("unpaired reference initialization")
            initial_reference = current
            if model.fusion is not None:
                current = tensor_digest(model.fusion.state_dict())
                if common_initial is not None and current != common_initial:
                    raise ValueError("unpaired local-head/gate initialization")
                common_initial = current
            model.eval()
            with torch.no_grad():
                if initial_logits is None:
                    initial_logits = model.reference(source.model.tensor_images(images, "cuda"))[0].cpu()
                before = model(source.model.tensor_images(images, "cuda"),
                               torch.from_numpy(patch_prior(priors)).cuda())[0].cpu()
            torch.testing.assert_close(initial_logits, before, rtol=0, atol=0)
            model.train()
            optimizer = optimizer_for(model)
            torch.cuda.reset_peak_memory_stats()
            times, logs, traces = [], [], []
            for step in range(3):
                guard()
                torch.cuda.synchronize()
                begun = time.monotonic()
                batch = paired_batch(rows, load_rgb, load_prior, source.augmentation.FASAug(),
                                     source, SEED, 1, step)
                traces.append(batch.trace)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(source.model.tensor_images(batch.images, "cuda"),
                                torch.from_numpy(batch.priors).cuda())
                values = source.model.losses(
                    outputs, torch.from_numpy(batch.labels).cuda(), torch.from_numpy(batch.domains).cuda(),
                    torch.from_numpy(batch.patch_labels).cuda(), tail=False)
                values["total"].backward()
                check_gradients(model, source)
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.monotonic()-begun
                if step:
                    times.append(elapsed)
                    logs.append({name: value.item() for name, value in values.items()})
            if common_traces is not None and traces != common_traces:
                raise ValueError("cross-arm pixel/label/geometry/PDA pairing differs")
            common_traces = traces
            model.eval()
            torch.cuda.synchronize()
            begun = time.monotonic()
            with torch.no_grad():
                baseline = model(source.model.tensor_images(images, "cuda"),
                                 torch.from_numpy(patch_prior(priors)).cuda())[0].cpu()
            torch.cuda.synchronize()
            prediction_seconds = time.monotonic()-begun
            checkpoint = folder/"model.pth"
            begun = time.monotonic()
            torch.save({"state_dict": model.state_dict(), "arm": arm,
                        "settings_sha256": object_digest(SETTINGS)}, checkpoint)
            save_seconds = time.monotonic()-begun
            record = {"arm": arm, "measured_step_seconds": times, "measured_losses": logs,
                      "inference_images": 6, "inference_seconds": prediction_seconds,
                      "save_seconds": save_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
                      "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
                      "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            del optimizer, outputs, values, model
            gc.collect()
            torch.cuda.empty_cache()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restored = SpatialReference(source.model.Reference(initialize=False), arm).cuda().eval()
            restored.load_state_dict(state["state_dict"], strict=True)
            with torch.no_grad():
                replay = restored(source.model.tensor_images(images, "cuda"),
                                  torch.from_numpy(patch_prior(priors)).cuda())[0].cpu()
            torch.testing.assert_close(baseline, replay, rtol=0, atol=0)
            del state, restored
            gc.collect()
            torch.cuda.empty_cache()
            script = Path(__file__).resolve().parents[2]/"scripts/spatial_dino.py"
            begun = time.monotonic()
            completed = subprocess.run(
                [sys.executable, str(script), "_replay", "--fas-root", str(fas_root),
                 "--checkpoint", str(checkpoint), "--checkpoint-sha256", digest(checkpoint)],
                check=True, capture_output=True, text=True, timeout=90,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"})
            import json
            independent = json.loads(completed.stdout)
            torch.testing.assert_close(baseline, torch.tensor(independent["logits"]), rtol=0, atol=0)
            record.update({"independent_replay_seconds": time.monotonic()-begun,
                           "same_process_replay_exact": True, "independent_replay_exact": True,
                           "initial_reference_exact": True})
            results.append(record)
            checkpoint.unlink()
            print("SPATIAL_SYNTHETIC_ARM_COMPLETE", arm, flush=True)
    guard()
    write_json(output, {
        "status": "complete", "scope": "synthetic_engineering_only_not_PAD_results",
        "scientific_execution_authorized": False, "scientific_training_starts": 0,
        "PAD_image_decodes": 0, "target_evaluations": 0, "synthetic_optimizer_steps": 15,
        "synthetic_encoder_image_forwards": 606, "synthetic_checkpoints_retained": 0,
        "initial_reference_sha256": initial_reference, "common_fusion_sha256": common_initial,
        "paired_batch_trace_sha256": object_digest(common_traces), "results": results,
        "caps": PROBE_CAPS, "wall_seconds": time.monotonic()-started,
        "gpu": torch.cuda.get_device_name(0), "precision": "FP32_no_AMP_no_TF32",
        "code_sha256": code_pins(Path(__file__).resolve().parents[2]),
        "runtime": {**runtime(), "Pillow": importlib_pillow_version()},
        "excluded_costs": ["PAD_file_IO_and_decode", "full_epochs", "full_role_inference",
                           "full_20_checkpoint_verification", "bootstrap_and_closure"],
    })


def importlib_pillow_version() -> str:
    from PIL import __version__
    return __version__


def estimate_budget(inventory: list[dict[str, Any]], probe: dict[str, Any]) -> dict[str, Any]:
    if (probe["status"] != "complete" or probe["scientific_training_starts"] != 0
            or probe["PAD_image_decodes"] != 0
            or {item["arm"] for item in probe["results"]} != set(ARMS)):
        raise ValueError("successful five-arm synthetic probe required")
    measurements = {item["arm"]: item for item in probe["results"]}
    raw_training = sum(item["maximum_optimizer_steps"]*
                       max(measurements[item["arm"]]["measured_step_seconds"]) for item in inventory)
    raw_validation = sum(SETTINGS["epochs"]*item["validation_usable"]*
                         measurements[item["arm"]]["inference_seconds"]/6 for item in inventory)
    checkpoints = sum(measurements[item["arm"]]["checkpoint_bytes"] for item in inventory)
    conservative_seconds = 3*(raw_training+raw_validation)+6*3600
    return {"raw_max_epoch_synthetic_training_seconds": raw_training,
            "raw_max_epoch_synthetic_validation_seconds": raw_validation,
            "planning_seconds_with_3x_factor_plus_6h_reserve": conservative_seconds,
            "proposed_wall_cap_seconds": PROPOSED_LIMITS["max_seconds"],
            "estimated_checkpoint_bytes": checkpoints,
            "maximum_optimizer_steps": sum(item["maximum_optimizer_steps"] for item in inventory),
            "fits_within_proposed_wall_estimate": conservative_seconds <= PROPOSED_LIMITS["max_seconds"],
            "is_guaranteed_upper_bound": False,
            "limitation": "short_synthetic_measurement_excludes_real_IO_thermal_and_shared_device_variance"}


def prepare(fas_root: Path, output: Path, probe_path: Path) -> Path:
    require_private_output(output/"plan.json")
    if output.exists():
        raise FileExistsError("new spatial preparation root required")
    begun = time.monotonic()
    repository = Path(__file__).resolve().parents[2]
    probe = read_json(probe_path)
    supervision = read_json(probe_path.parent/"supervision.json")
    if (not supervision["completed"] or probe["code_sha256"] != code_pins(repository)
            or probe["runtime"] != {**runtime(), "Pillow": importlib_pillow_version()}):
        raise ValueError("successful supervised probe for exact current code/runtime required")
    services, rows, partitions = inputs(fas_root)
    files = {str(probe_path): digest(probe_path),
             str(probe_path.parent/"supervision.json"): digest(probe_path.parent/"supervision.json")}
    source = source_modules(fas_root)
    assets = source.augmentation.assets.verify()

    def bind(path: Path, expected: str | None = None) -> None:
        if time.monotonic()-begun > 1800:
            raise TimeoutError("metadata preparation 30-minute cap")
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError(f"input commitment changed: {path.name}")
        files[str(path)] = value

    prep = services["prepare_paper319"]
    bind(prep.MANIFEST, prep.MANIFEST_SHA)
    bind(prep.OUTPUT/"partitions.json")
    bind(source.model.CHECKPOINT, "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71")
    for path in sorted((fas_root/"scripts/paper329").glob("*.py")):
        bind(path)
    bind(fas_root/"scripts/paper327/objective.py")
    for name in (*assets["files"], "receipt.json"):
        bind(source.augmentation.assets.DIRECTORY/name)
    entries, failures = [], Counter()
    for row in rows:
        key = f"{row['dataset']}/{row['video_id']}"
        if row["detector_status"] != "success":
            entries.append({"key": key, "status": "historical_failure"})
            continue
        bind(Path(row["record_path"]), row["record_sha256"])
        crop = row["crops"]["224"]
        bind(Path(crop["path"]), crop["sha256"])
        try:
            points = points224(transaction_for(row))
        except RegionSupportError as error:
            failures[f"{row['dataset']}:{error}"] += 1
            entries.append({"key": key, "status": "shared_geometry_failure", "reason": str(error)})
        else:
            entries.append({"key": key, "status": "usable", "landmarks224": points.tolist()})
    usable = {item["key"] for item in entries if item["status"] == "usable"}
    contexts = [item for item in partitions["contexts"] if item["seed"] == SEED]
    if len(contexts) != 4:
        raise ValueError("exact four-target one-seed inventory required")
    inventory = []
    for context in contexts:
        counts = {role: sum(key in usable for key in context[role])
                  for role in ("fitting", "validation", "calibration", "evaluation")}
        indexed = {f"{row['dataset']}/{row['video_id']}": row for row in rows}
        for role in ("fitting", "validation", "calibration"):
            pool = [indexed[key] for key in context[role] if key in usable]
            strata = {(row["dataset"], row["binary_label"]) for row in pool}
            if len(strata) != 6 or any(row["dataset"] == context["target"] for row in pool):
                raise ValueError("source-only domain/class support required after shared failures")
        for arm in ARMS:
            inventory.append({"target": context["target"], "arm": arm, "seed": SEED,
                              **{f"{role}_usable": value for role, value in counts.items()},
                              "maximum_optimizer_steps": ((counts["fitting"]+31)//32)*SETTINGS["epochs"]})
    estimate = estimate_budget(inventory, probe)
    plan = {"study": "spatial_dino_preparation_v1", "scientific_execution_authorized": False,
            "executable_scientific_runner_available": True, "execution_runner_sealed": False,
            "policy": POLICY, "settings": SETTINGS,
            "code_sha256": code_pins(repository), "files": files, "entries": entries,
            "contexts": contexts, "inventory": inventory, "proposed_limits": PROPOSED_LIMITS,
            "budget_estimate": estimate, "new_shared_geometry_failures": dict(failures),
            "counts": dict(Counter(item["status"] for item in entries)),
            "PAD_image_decodes": 0, "scientific_training_starts": 0, "crop_files_hashed": 7474,
            "preparation_wall_seconds": time.monotonic()-begun}
    path = output/"plan.json"
    write_json(path, plan)
    summary = {key: value for key, value in plan.items() if key not in ("files", "entries", "contexts")}
    write_json(output/"summary.json", {**summary, "plan_sha256": digest(path), "bound_file_count": len(files),
                                      "synthetic_probe_sha256": digest(probe_path)})
    return path
