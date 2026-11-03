"""Matched source-only adapter pilot; never a PAD operating-point evaluation."""

from __future__ import annotations

from collections import Counter
import copy
import gc
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from .data import Video, load_manifest
from .dataset import ClipCache, balanced_draws
from .gpu_probe import tensor_digest
from .io import digest, read_json, require_private_output, write_json
from .models import build_model
from .scoring import residual_profile
from .source_preparation import selected_sources
from .training import check_source_pools, install_trainable, sampling_commitment, train, validation_metrics

ARMS = ("pretrained_spatial_adapter", "pretrained_temporal")
GPU_QUERY = Path("/usr/lib/wsl/lib/nvidia-smi")


def device_memory_bytes() -> int:
    result = subprocess.run(
        [str(GPU_QUERY), "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    value = result.stdout.strip()
    if not value.isdecimal():
        raise ValueError("finite device-wide GPU memory telemetry required")
    return int(value) * 1024**2


def public_validation(metrics: dict[str, Any]) -> dict[str, Any]:
    return {name: metrics[name] for name in ("macro_loss", "by_domain")}


def pilot_config(template: dict[str, Any], bundle: Path, bundle_sha: str) -> dict[str, Any]:
    config = copy.deepcopy(template)
    if config["target"] != "OULU-NPU" or config["clip"]["duration"] != 2 or config["clip"]["fps"] != 8:
        raise ValueError("pilot requires OULU-held-out 2s/8FPS contract")
    config["seed"] = 20261103
    config["execution"] = {name: name in {"source_training", "source_scoring"}
                           for name in config["execution"]}
    config["model"].update(local_bundle=str(bundle.resolve()), bundle_manifest_sha256=bundle_sha,
                           adapter_channels=32)
    config["model"]["candidate"]["weights_acquired"] = True
    config["model"]["candidate"]["limitations"][-1] = "Verified local bundle; no pilot download."
    config["training"].update(effective_batch=2, microbatch=1, max_epochs=2, minimum_epochs=2,
                               patience=2, warmup_epochs=0, numeric="float32")
    return config


def build_plan(
    manifest: Path, manifest_sha: str, preparation_plan: Path, bundle: Path, bundle_sha: str,
    template: Path, output: Path,
) -> Path:
    require_private_output(output / "placeholder")
    if output.exists():
        raise ValueError("new adapter-pilot root required")
    if digest(manifest) != manifest_sha:
        raise ValueError("prepared manifest commitment mismatch")
    completion = read_json(preparation_plan.parent / "completion.json")
    supervision = read_json(preparation_plan.parent / "supervision.json")
    preparation_sha = digest(preparation_plan)
    if (completion["prepared_manifest_sha256"] != manifest_sha
            or completion["plan_sha256"] != preparation_sha
            or not completion["completed"] or not supervision["completed"]
            or supervision["plan_sha256"] != preparation_sha):
        raise ValueError("completed source-preparation evidence required")
    videos, provenance = load_manifest(manifest)
    if provenance["preparation_plan_sha256"] != preparation_sha:
        raise ValueError("prepared population lineage mismatch")
    sources = selected_sources(videos)
    fitting, validation, skipped = check_source_pools(sources, "OULU-NPU")
    bundle_manifest = read_json(bundle / "manifest.json", bundle_sha)
    if (bundle_manifest.get("rights_reviewed") is not True
            or set(bundle_manifest["files"]) != {"config.json", "scheduler_config.json", "model.safetensors"}):
        raise ValueError("verified local safe bundle required")
    for name, sha in bundle_manifest["files"].items():
        if digest(bundle / name) != sha:
            raise ValueError("bundle file commitment mismatch")
    config = pilot_config(read_json(template), bundle, bundle_sha)
    steps = 2 * math.ceil(len(fitting) / 2)
    if not 300 <= steps <= 500:
        raise ValueError("two-epoch pilot must remain within 300-500 steps per arm")
    config["budget"] = {"max_seconds": 2700, "max_optimizer_steps": steps,
                        "max_output_bytes": 2 * 1024**3}
    schedule = []
    for epoch in range(2):
        draws = balanced_draws(fitting, config["seed"], epoch)
        schedule.append({"epoch": epoch, "sha256": sampling_commitment(draws, config["seed"], epoch),
                         "draws_by_domain": dict(Counter(v.dataset for v in draws))})
    project = Path(__file__).resolve().parents[2]
    plan = {
        "schema_version": 1, "scope": "source_only_adapter_pilot_not_PAD_evaluation",
        "manifest": str(manifest.resolve()), "manifest_sha256": manifest_sha,
        "preparation_plan": str(preparation_plan.resolve()), "preparation_plan_sha256": preparation_sha,
        "preparation_completion_sha256": digest(preparation_plan.parent / "completion.json"),
        "preparation_supervision_sha256": digest(preparation_plan.parent / "supervision.json"),
        "bundle": str(bundle.resolve()), "bundle_manifest_sha256": bundle_sha,
        "config": config, "schedule": schedule, "steps_per_arm": steps,
        "fitting_originals": len(fitting), "validation_originals": len(validation),
        "source_original_denominator": len(sources), "retained_failures": skipped,
        "implementation_sha256": {str(p.relative_to(project)): digest(p)
                                  for directory in ("src", "scripts")
                                  for p in sorted((project / directory).rglob("*.py"))},
        "budget": {"max_seconds": 2700, "max_gpu_bytes": 8 * 1024**3,
                   "max_rss_bytes": 8 * 1024**3, "max_output_bytes": 2 * 1024**3},
    }
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def checked_inputs(path: Path, sha: str) -> tuple[dict[str, Any], list[Video]]:
    plan = read_json(path, sha)
    require_private_output(path)
    project = Path(__file__).resolve().parents[2]
    for name, expected in plan["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("pilot implementation changed after plan")
    if digest(Path(plan["manifest"])) != plan["manifest_sha256"]:
        raise ValueError("pilot population changed after plan")
    previous = Path(plan["preparation_plan"])
    read_json(previous, plan["preparation_plan_sha256"])
    read_json(previous.parent / "completion.json", plan["preparation_completion_sha256"])
    read_json(previous.parent / "supervision.json", plan["preparation_supervision_sha256"])
    if digest(Path(plan["bundle"]) / "manifest.json") != plan["bundle_manifest_sha256"]:
        raise ValueError("pilot bundle changed after plan")
    permissions = plan["config"]["execution"]
    if plan["budget"] != {"max_seconds": 2700, "max_gpu_bytes": 8 * 1024**3,
                          "max_rss_bytes": 8 * 1024**3, "max_output_bytes": 2 * 1024**3}:
        raise ValueError("pilot caps differ from explicit approval")
    if {name for name, value in permissions.items() if value} != {"source_training", "source_scoring"}:
        raise PermissionError("only bonafide source training/validation is authorized")
    videos, _ = load_manifest(Path(plan["manifest"]))
    sources = selected_sources(videos)
    fitting, validation, _ = check_source_pools(sources, "OULU-NPU")
    if len(fitting) != plan["fitting_originals"] or len(validation) != plan["validation_originals"]:
        raise ValueError("pilot pool size changed")
    return plan, sources


def configure_gpu(plan: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ValueError("approved pilot requires CUDA; no CPU substitute")
    free, total = torch.cuda.mem_get_info()
    allocator_limit = min(plan["budget"]["max_gpu_bytes"] - 1024**3, free - 1024**3)
    if allocator_limit < 4 * 1024**3:
        raise MemoryError("insufficient free shared GPU memory")
    torch.cuda.set_per_process_memory_fraction(allocator_limit / total)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    return {"name": torch.cuda.get_device_properties(0).name, "initial_free_bytes": free,
            "allocator_limit_bytes": allocator_limit, "numeric": "FP32_no_AMP_no_TF32",
            "bitwise_training_determinism_claimed": False}


def representatives(validation: list[Video]) -> list[Video]:
    return [next(v for v in sorted(validation, key=lambda v: v.key) if v.dataset == domain)
            for domain in sorted({v.dataset for v in validation})]


def run_pilot(path: Path, sha: str, script: Path) -> None:
    plan, sources = checked_inputs(path, sha)
    root = path.parent
    if (root / "started.json").exists():
        raise ValueError("pilot already attempted; no automatic retry")
    started = time.monotonic()
    deadline = started + plan["budget"]["max_seconds"] - 20
    write_json(root / "started.json", {"plan_sha256": sha})
    device = configure_gpu(plan)
    write_json(root / "device.json", device)
    _, validation, _ = check_source_pools(sources, "OULU-NPU")
    cache = ClipCache(256)
    arms, baseline = [], None
    initial_profiles = None

    def check_deadline(module, args):
        if time.monotonic() >= deadline:
            raise TimeoutError("shared pilot wall-clock budget exhausted")

    for variant in ARMS:
        config = copy.deepcopy(plan["config"])
        config["model"]["variant"] = variant
        torch.manual_seed(config["seed"])
        model = build_model(config).cuda()
        if model.adapter is None or not model.freeze_backbone:
            raise ValueError("pilot requires a trainable adapter and frozen backbone")
        model.backbone_frame_batch = 4
        model.register_forward_pre_hook(check_deadline)
        torch.cuda.reset_peak_memory_stats()
        backbone_sha = tensor_digest(model.backbone.state_dict())
        initial_adapter = tensor_digest(model.adapter.state_dict())
        if baseline is None:
            metrics = validation_metrics(model, cache, validation, torch.device("cuda"), config["seed"])
            write_json(root / "baseline-private.json", metrics)
            baseline = public_validation(metrics)
            chosen = {v.key for v in representatives(validation)}
            initial_profiles = {r["key"]: r["profile"] for r in metrics["records"] if r["key"] in chosen}
        else:
            if initial_profiles is None:
                raise ValueError("zero-init baseline evidence missing")
            for video in representatives(validation):
                profile = residual_profile(model, cache.get(video)[None].cuda(), [video.key],
                                           alpha_bars=[0.9, 0.5, 0.1], draws=2, seed=config["seed"])
                torch.testing.assert_close(profile.cpu(), torch.tensor([initial_profiles[video.key]]),
                                           rtol=1e-5, atol=1e-6)
        arm_started = time.monotonic()
        output = root / variant
        write_json(root / f"started-{variant}.json", {"plan_sha256": sha, "variant": variant})
        paused = train(model, sources, config, output, device="cuda", stop_after_epochs=1)
        if paused.get("paused_for_resume") is not True or paused["complete"]:
            raise ValueError("pilot did not produce the declared incomplete epoch checkpoint")
        incomplete = torch.load(output / "last.pt", weights_only=True, map_location="cpu")
        if incomplete["complete"] or incomplete["next_epoch"] != 1:
            raise ValueError("invalid epoch-boundary resume checkpoint")
        print({"variant": variant, "stage": "resuming", "steps": paused["steps"]}, flush=True)
        result = train(model, sources, config, output, device="cuda", resume=output / "last.pt")
        last = torch.load(output / "last.pt", weights_only=True, map_location="cpu")
        best = torch.load(output / "best.pt", weights_only=True, map_location="cpu")
        if (not result["complete"] or not last["complete"] or result["steps"] != plan["steps_per_arm"]
                or [h["sampling_sha256"] for h in last["history"]] != [s["sha256"] for s in plan["schedule"]]):
            raise ValueError("completed pilot steps/sampling differ from locked plan")
        if tensor_digest(model.backbone.state_dict()) != backbone_sha:
            raise ValueError("adapter training modified frozen backbone")
        if tensor_digest(model.adapter.state_dict()) == initial_adapter:
            raise ValueError("adapter parameters did not change")
        arms.append({
            "variant": variant, "steps": result["steps"], "epoch_boundary_resume_verified": True,
            "history": last["history"], "best_epoch": last["best_epoch"],
            "best_validation": public_validation(best["validation"]),
            "last_validation": public_validation(last["validation"]),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_backbone_unchanged": True, "frozen_backbone_sha256": backbone_sha,
            "adapter_changed": True, "training_and_validation_seconds": time.monotonic() - arm_started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "last_checkpoint_sha256": digest(output / "last.pt"),
            "best_checkpoint_sha256": digest(output / "best.pt"),
        })
        write_json(root / f"result-{variant}.json", arms[-1])
        print({"variant": variant, "stage": "completed", "steps": result["steps"],
               "best_validation_loss": best["best_loss"]}, flush=True)
        del model, last, best, incomplete
        gc.collect()
        torch.cuda.empty_cache()
    training = {"plan_sha256": sha, "baseline": baseline, "arms": arms, "device": device}
    write_json(root / "training-result.json", training)
    subprocess.run(
        [sys.executable, str(script), "replay", "--plan", str(path), "--plan-sha256", sha,
         "--training-result-sha256", digest(root / "training-result.json")],
        check=True, timeout=max(1, deadline - time.monotonic()),
    )
    write_json(root / "completion.json", {
        "completed": True, "plan_sha256": sha,
        "training_result_sha256": digest(root / "training-result.json"),
        "replay_sha256": digest(root / "replay.json"), "wall_seconds": time.monotonic() - started,
    })


def replay_pilot(path: Path, sha: str, training_sha: str) -> None:
    plan, sources = checked_inputs(path, sha)
    root = path.parent
    training = read_json(root / "training-result.json", training_sha)
    configure_gpu(plan)
    _, validation, _ = check_source_pools(sources, "OULU-NPU")
    cache, errors = ClipCache(256), {}
    for arm in training["arms"]:
        variant = arm["variant"]
        checkpoint = root / variant / "best.pt"
        if digest(checkpoint) != arm["best_checkpoint_sha256"]:
            raise ValueError("saved best checkpoint commitment mismatch")
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        model = build_model(state["config"]).cuda().eval()
        model.backbone_frame_batch = 4
        install_trainable(model, state["ema"])
        expected = {r["key"]: r["profile"] for r in state["validation"]["records"]}
        maximum = 0.0
        for video in representatives(validation):
            profile = residual_profile(model, cache.get(video)[None].cuda(), [video.key],
                                       alpha_bars=[0.9, 0.5, 0.1], draws=2, seed=plan["config"]["seed"])
            reference = torch.tensor([expected[video.key]])
            torch.testing.assert_close(profile.cpu(), reference, rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float((profile.cpu() - reference).abs().max()))
        if tensor_digest(model.backbone.state_dict()) != arm["frozen_backbone_sha256"]:
            raise ValueError("reloaded backbone differs from trained frozen backbone")
        errors[variant] = {"validation_originals": 3, "max_absolute_profile_error": maximum,
                           "rtol": 1e-5, "atol": 1e-6, "passed": True}
        del model, state
        gc.collect()
        torch.cuda.empty_cache()
    write_json(root / "replay.json", {"plan_sha256": sha, "training_result_sha256": training_sha,
                                    "separate_process": True, "arms": errors, "completed": True})


def summarize(path: Path, supervision: dict[str, Any]) -> dict[str, Any]:
    plan = read_json(path)
    root = path.parent
    completion = read_json(root / "completion.json") if (root / "completion.json").exists() else None
    training, replay = None, None
    if completion is not None:
        if completion["plan_sha256"] != digest(path):
            raise ValueError("pilot completion/plan mismatch")
        training = read_json(root / "training-result.json", completion["training_result_sha256"])
        replay = read_json(root / "replay.json", completion["replay_sha256"])
        if (training["plan_sha256"] != digest(path) or replay["plan_sha256"] != digest(path)
                or replay["training_result_sha256"] != completion["training_result_sha256"]):
            raise ValueError("pilot result/plan mismatch")
    completed_arms = [read_json(root / f"result-{variant}.json") for variant in ARMS
                      if (root / f"result-{variant}.json").exists()]
    return {
        "schema_version": 1, "scope": plan["scope"],
        "completed": bool(supervision["completed"] and completion is not None and completion["completed"]
                          and replay is not None and replay["completed"]),
        "selection_plan_sha256": digest(path), "prepared_manifest_sha256": plan["manifest_sha256"],
        "preparation_plan_sha256": plan["preparation_plan_sha256"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "implementation_sha256": plan["implementation_sha256"], "budget": plan["budget"],
        "source_original_denominator": plan["source_original_denominator"],
        "fitting_originals": plan["fitting_originals"], "validation_originals": plan["validation_originals"],
        "retained_failure_count": sum(plan["retained_failures"].values()),
        "matched_schedule": plan["schedule"], "supervision": supervision,
        "training": training, "replay": replay,
        "completed_arms": completed_arms,
        "scientific_source_training_runs": sum((root / f"started-{variant}.json").exists()
                                               for variant in ARMS),
        "raw_video_decodes": 0, "attack_model_forwards": 0, "target_model_forwards": 0,
        "calibration_fitted": False, "PAD_accuracy_claimed": False,
        "automatic_retries": 0, "checkpoints_and_individual_scores_private": True,
    }
