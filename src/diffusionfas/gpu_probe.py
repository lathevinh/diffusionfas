"""GPU engineering measurements on synthetic tensors, never PAD examples."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import time
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .io import digest, read_json, require_private_output, write_json
from .models import local_pretrained_model
from .scoring import residual_profile

ARMS = {
    "frozen": {"temporal": False, "temporal_kernel": 1},
    "spatial_adapter": {"temporal": True, "temporal_kernel": 1},
    "temporal_adapter": {"temporal": True, "temporal_kernel": 3},
}


def configure_gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no CPU performance substitute")
    free, total = torch.cuda.mem_get_info()
    allowed = min(int(total * 0.65), free - 2 * 1024**3, 12 * 1024**3)
    if allowed < 4 * 1024**3:
        raise MemoryError("insufficient free shared GPU memory for bounded probe")
    torch.cuda.set_per_process_memory_fraction(allowed / total)
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    torch.cuda.manual_seed_all(20261103)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    return {
        "name": torch.cuda.get_device_properties(0).name, "total_bytes": total,
        "initial_free_bytes": free, "own_allocator_limit_bytes": allowed,
        "numeric": "FP32_no_AMP_no_TF32", "training_bitwise_determinism_claimed": False,
    }


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    checksum = hashlib.sha256()
    for name in sorted(tensors):
        checksum.update(name.encode())
        checksum.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def run_gpu_probe(bundle: Path, bundle_sha: str, output: Path) -> dict[str, Any]:
    require_private_output(output / "placeholder")
    if output.exists():
        raise ValueError("new synthetic GPU probe output directory required")
    output.mkdir(parents=True)
    device = configure_gpu()
    generator = torch.Generator(device="cpu").manual_seed(20261103)
    clean = torch.rand((1, 16, 3, 256, 256), generator=generator).mul(2).sub(1)
    epsilon = torch.randn(clean.shape, generator=generator)
    timestep = torch.tensor([500], dtype=torch.long)
    save_file({"clean": clean, "epsilon": epsilon, "timestep": timestep},
              str(output / "synthetic-input.safetensors"))
    clean, epsilon, timestep = clean.cuda(), epsilon.cuda(), timestep.cuda()
    arms = []
    baseline = None
    for name, options in ARMS.items():
        torch.manual_seed(20261103)
        model = local_pretrained_model(bundle, bundle_sha, **options).cuda()
        model.backbone_frame_batch = 4
        torch.cuda.reset_peak_memory_stats()
        initial_backbone = tensor_digest(model.backbone.state_dict())
        noisy = model.scheduler.add_noise(clean, epsilon, timestep)
        model.eval()
        with torch.no_grad():
            initial = model(noisy, timestep)
        if baseline is None:
            baseline = initial.cpu()
        identity_error = float((initial.cpu() - baseline).abs().max())
        if identity_error > 1e-6:
            raise ValueError("zero-init adapter does not preserve frozen pretrained prediction")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4) if parameters else None
        times, losses, gradient_norms = [], [], []
        for iteration in range(4):
            model.train(optimizer is not None)
            torch.cuda.synchronize()
            started = time.monotonic()
            if optimizer is None:
                with torch.no_grad():
                    prediction = model(noisy, timestep)
                    loss = (prediction - epsilon).square().mean()
            else:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(noisy, timestep)
                loss = (prediction - epsilon).square().mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                gradient_norms.append(float(norm))
                optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite synthetic probe loss")
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            losses.append(float(loss.detach()))
            if iteration:
                times.append(elapsed)
        if tensor_digest(model.backbone.state_dict()) != initial_backbone:
            raise ValueError("synthetic adapter update changed frozen backbone")
        if optimizer is not None and not any(g > 0 for g in gradient_norms):
            raise ValueError("synthetic adapter did not receive any nonzero gradients")
        model.eval()
        with torch.no_grad():
            expected = model(noisy, timestep)
        save_file({"prediction": expected.cpu()}, str(output / f"{name}-reference.safetensors"))
        adapter_sha = None
        if model.adapter is not None:
            checkpoint = output / f"{name}-adapter.safetensors"
            save_file({key: value.detach().cpu().contiguous()
                       for key, value in model.adapter.state_dict().items()}, str(checkpoint))
            adapter_sha = digest(checkpoint)
        torch.cuda.synchronize()
        scoring_started = time.monotonic()
        profile = residual_profile(model, clean, ["synthetic-probe"], alpha_bars=[0.9, 0.5, 0.1],
                                   draws=2, seed=20261103)
        torch.cuda.synchronize()
        scoring_seconds = time.monotonic() - scoring_started
        report = {
            "arm": name, "input_shape": [1, 16, 3, 256, 256],
            "backbone_frame_microbatch": 4, "clip_microbatch": 1,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "frozen_backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
            "zero_init_max_absolute_difference": identity_error,
            "frozen_backbone_unchanged": True, "synthetic_optimizer_steps": 4 if optimizer else 0,
            "warmup_iterations": 1, "measured_iterations": 3,
            "measured_step_seconds": times, "mean_step_seconds": sum(times) / len(times),
            "losses": losses, "gradient_norms": gradient_norms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "six_evaluation_scoring_seconds": scoring_seconds,
            "synthetic_residual_profile": profile.cpu().tolist(),
            "adapter_checkpoint_sha256": adapter_sha,
        }
        write_json(output / f"{name}.json", report)
        arms.append(report)
        del model, optimizer, parameters, initial, prediction, loss, noisy, expected, profile
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "schema_version": 1, "completed": True, "device": device, "arms": arms,
        "bundle_manifest_sha256": bundle_sha, "synthetic_inputs_only": True,
        "scientific_training_runs": 0, "source_video_model_forwards": 0,
        "target_evaluations": 0, "scope": "engineering_probe_not_PAD_performance",
    }
    write_json(output / "gpu-probe.json", result)
    return result


def replay_gpu_probe(bundle: Path, bundle_sha: str, probe: Path, output: Path) -> dict[str, Any]:
    configure_gpu()
    inputs = load_file(str(probe / "synthetic-input.safetensors"))
    clean, epsilon, timestep = inputs["clean"].cuda(), inputs["epsilon"].cuda(), inputs["timestep"].cuda()
    errors = {}
    for name, options in ARMS.items():
        model = local_pretrained_model(bundle, bundle_sha, **options).cuda().eval()
        model.backbone_frame_batch = 4
        if model.adapter is not None:
            receipt = read_json(probe / f"{name}.json")
            checkpoint = probe / f"{name}-adapter.safetensors"
            if digest(checkpoint) != receipt["adapter_checkpoint_sha256"]:
                raise ValueError("adapter checkpoint changed before independent replay")
            model.adapter.load_state_dict(load_file(str(checkpoint)), strict=True)
        with torch.no_grad():
            predicted = model(model.scheduler.add_noise(clean, epsilon, timestep), timestep).cpu()
        expected = load_file(str(probe / f"{name}-reference.safetensors"))["prediction"]
        error = float((predicted - expected).abs().max())
        if not torch.allclose(predicted, expected, rtol=1e-5, atol=1e-6):
            raise ValueError("separate-process pretrained/adapter replay failed")
        errors[name] = error
        del model, predicted, expected
        gc.collect()
        torch.cuda.empty_cache()
    result = {"completed": True, "separate_process": True, "maximum_absolute_errors": errors,
              "rtol": 1e-5, "atol": 1e-6, "synthetic_inputs_only": True}
    write_json(output, result)
    return result


def verify_native_microbatch(bundle: Path, bundle_sha: str, probe: Path, output: Path) -> dict[str, Any]:
    device = configure_gpu()
    inputs = load_file(str(probe / "synthetic-input.safetensors"))
    clean, epsilon, timestep = inputs["clean"].cuda(), inputs["epsilon"].cuda(), inputs["timestep"].cuda()
    model = local_pretrained_model(bundle, bundle_sha, temporal=False).cuda().eval()
    noisy = model.scheduler.add_noise(clean, epsilon, timestep)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        native = model(noisy, timestep)
        model.backbone_frame_batch = 4
        microbatched = model(noisy, timestep)
    difference = float((native - microbatched).abs().max())
    if difference > 1e-4 or not torch.allclose(native, microbatched, rtol=1e-4, atol=1e-5):
        raise ValueError("actual pretrained U-Net frame microbatch changes predictions beyond FP32 tolerance")
    result = {
        "completed": True, "device": device, "input_shape": list(noisy.shape),
        "compared_frame_batches": [16, 4], "max_absolute_difference": difference,
        "rtol": 1e-4, "atol": 1e-5, "maximum_absolute_difference_limit": 1e-4,
        "peak_allocated_bytes_including_native_16_frame_forward": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes_including_native_16_frame_forward": torch.cuda.max_memory_reserved(),
        "bundle_manifest_sha256": bundle_sha, "synthetic_inputs_only": True,
        "scientific_training_runs": 0, "source_video_model_forwards": 0,
    }
    write_json(output, result)
    return result
