"""Explicit, bounded public-weight acquisition and lossless safe conversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any
import urllib.request

import torch
from diffusers import DDPMScheduler, UNet2DModel
from safetensors.torch import save_file

from .io import digest, read_json, require_private_output, write_json

MODEL_ID = "google/ddpm-celebahq-256"
REVISION = "cd5c944777ea2668051904ead6cc120739b86c4d"
WEIGHT_SHA = "efff89712093ad060ce99d9b461bbe542b49d8dd4ce30f23fe5761dca292361d"
WEIGHT_BYTES = 454853117
TEXT_BLOBS = {
    "config.json": "21be917d8db8da69de6145db1ed5e5d45b5086da",
    "scheduler_config.json": "2b0d557d4eb7e3cf63f8e9c610e507601f58ab36",
    "README.md": "7d52a0a253cf339e68b2ac1eebb504d13022ba92",
}


def download(
    url: str, output: Path, *, maximum_bytes: int, deadline: float,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if not url.startswith("https://huggingface.co/"):
        raise ValueError("only the declared public Hugging Face origin is supported")
    temporary = output.with_suffix(output.suffix + ".partial")
    if output.exists() or temporary.exists():
        raise ValueError("download outputs must be new; no automatic retry/resume")
    request = urllib.request.Request(url, headers={"User-Agent": "diffusionfas-research-probe/0.1"})
    total, checksum = 0, hashlib.sha256()
    with urllib.request.urlopen(request, timeout=30) as response, temporary.open("xb") as stream:
        if not response.url.startswith("https://"):
            raise ValueError("insecure model redirect prohibited")
        while chunk := response.read(1024 * 1024):
            if time.monotonic() > deadline:
                raise TimeoutError("public weight acquisition deadline")
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("download exceeds declared byte cap")
            checksum.update(chunk)
            stream.write(chunk)
    observed = checksum.hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError("download differs from pinned upstream LFS SHA-256")
    temporary.rename(output)
    return {"bytes": total, "sha256": observed}


def convert_state(
    original: dict[str, torch.Tensor], expected: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    converted, renames, reshapes = {}, 0, 0
    if not isinstance(original, dict):
        raise ValueError("plain tensor state dict required; no arbitrary pickle execution")
    for key, tensor in original.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint must contain only named tensors")
        mapped = key
        for old, new in ((".query.", ".to_q."), (".key.", ".to_k."),
                         (".value.", ".to_v."), (".proj_attn.", ".to_out.0.")):
            if ".attentions." in mapped:
                mapped = mapped.replace(old, new)
        if mapped not in expected or mapped in converted:
            raise ValueError(f"unrecognized/colliding legacy state key: {mapped}")
        if mapped != key:
            renames += 1
        target_shape = expected[mapped].shape
        if tensor.shape != target_shape:
            if (
                ".attentions." not in mapped or tensor.ndim != 4 or tensor.shape[-2:] != (1, 1)
                or tensor.shape[:2] != target_shape
            ):
                raise ValueError(f"unsupported state tensor shape conversion: {mapped}")
            tensor = tensor.reshape(target_shape)
            reshapes += 1
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite upstream tensor")
        converted[mapped] = tensor.detach().cpu().contiguous()
    if set(converted) != set(expected):
        raise ValueError("converted checkpoint does not cover every native model key")
    return converted, {"renamed_attention_tensors": renames, "reshaped_1x1_attention_tensors": reshapes}


def acquire_bundle(root: Path, *, max_seconds: float = 900) -> dict[str, Any]:
    require_private_output(root / "placeholder")
    if root.exists():
        raise ValueError("new acquisition directory required")
    root.mkdir(parents=True)
    deadline = time.monotonic() + max_seconds
    original, bundle = root / "upstream", root / "bundle"
    original.mkdir()
    bundle.mkdir()
    files = {}
    for name, blob in TEXT_BLOBS.items():
        files[name] = download(
            f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", original / name,
            maximum_bytes=16384, deadline=deadline,
        )
        content = (original / name).read_bytes()
        if hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() != blob:
            raise ValueError("upstream text differs from pinned Git blob")
    card = (original / "README.md").read_text()
    if "license: apache-2.0" not in card:
        raise ValueError("reviewed Apache-2.0 model-card declaration changed")
    files["diffusion_pytorch_model.bin"] = download(
        f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/diffusion_pytorch_model.bin",
        original / "diffusion_pytorch_model.bin", maximum_bytes=WEIGHT_BYTES,
        deadline=deadline, expected_sha256=WEIGHT_SHA,
    )
    if files["diffusion_pytorch_model.bin"]["bytes"] != WEIGHT_BYTES:
        raise ValueError("upstream weight byte count mismatch")
    torch.set_num_threads(2)
    model = UNet2DModel.from_config(read_json(original / "config.json"))
    state = torch.load(original / "diffusion_pytorch_model.bin", map_location="cpu", weights_only=True)
    converted, conversion = convert_state(state, model.state_dict())
    model.load_state_dict(converted, strict=True)
    if any(not torch.equal(model.state_dict()[name], value) for name, value in converted.items()):
        raise ValueError("converted model did not preserve upstream tensor values exactly")
    save_file(converted, str(bundle / "model.safetensors"))
    write_json(bundle / "config.json", dict(model.config))
    scheduler = DDPMScheduler.from_config(read_json(original / "scheduler_config.json"))
    write_json(bundle / "scheduler_config.json", dict(scheduler.config))
    manifest = {
        "upstream_repository": MODEL_ID, "upstream_revision": REVISION, "rights_reviewed": True,
        "rights_review_scope": "reviewed public Apache-2.0 model-card declaration for this private research probe, not legal certification or unrestricted dataset/commercial rights",
        "weights_redistributed": False,
        "files": {name: digest(bundle / name) for name in
                  ("config.json", "scheduler_config.json", "model.safetensors")},
    }
    write_json(bundle / "manifest.json", manifest)
    report = {
        "schema_version": 1, "completed": True, "upstream_repository": MODEL_ID,
        "upstream_revision": REVISION, "upstream_files": files,
        "conversion": conversion, "safe_load": "torch.load(weights_only=True)",
        "model_card_declared_license": "apache-2.0", "weights_redistributed": False,
        "rights_review_scope": manifest["rights_review_scope"],
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "native_resolution": model.config.sample_size,
        "native_noise_schedule": scheduler.config.beta_schedule,
        "unsafe_pickle_fallback": False, "scientific_training_runs": 0,
    }
    write_json(root / "acquisition.json", report)
    print(json.dumps({"completed": True, "upstream_weight_sha256": WEIGHT_SHA,
                      "bundle_manifest_sha256": report["bundle_manifest_sha256"]}), flush=True)
    return report
